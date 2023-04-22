import argparse
import collections
import json
import os
import pathlib
import sys
import cv2
sys.path.append("..")
sys.path.append("...")
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AdamW
# from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
from AOKVQA.dataset.clipcap_train import load_data,load_data_vqax
from model.Clipcap import load_model, ClipCapVQAModel
# from pycocotools.coco import COCO
# from evaluation.evaluation import compute_scores
from model.train_clipcap import ClipCocoDataset
import numpy as np
import utils.misc as utils
import random
from utils.load_data import load_aokvqa
from evaluation.cococaption.pycocoevalcap.eval import COCOEvalCap
# import evaluation.evaluation as  evaluation
# from accelerate import Accelerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', type=str, required=True, dest='log_dir')
    parser.add_argument('--aokvqa-dir', type=str, required=True, dest='aokvqa_dir')
    parser.add_argument('--train-features', type=str, required=True, dest='train_features')
    parser.add_argument('--val-features', type=str, required=True, dest='val_features')
    parser.add_argument('--pretrained-model', type=str, dest='pretrained_model')

    parser.add_argument('--prompt-with-choices', action='store_true', dest='prompt_with_choices')
    parser.add_argument('--generation-target', type=str, choices=['answer', 'rationale'], required=True, dest='generation_target')

    # Training hyperparams
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--bs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-5)
    # Model hyperparams
    parser.add_argument('--mapping', type=str, choices=['mlp', 'transformer'], required=True, dest='mapping_type')
    parser.add_argument('--prefix-length', type=int, dest='prefix_length')
    parser.add_argument('--clip-model-type', type=str, choices=('RN50x4', 'ViT-B/32'), dest='clip_model_type')
    parser.add_argument('--normalize-prefix', type=bool, dest='normalize_prefix')
    parser.add_argument('--num-layers', type=int, dest='num_layers')
    parser.add_argument('--seed', default=43, type=int)
    parser.add_argument('--finetune-gpt', action='store_true', dest='finetune_gpt')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    cfg = parser.parse_args()

    utils.init_distributed_mode(cfg)
    device = torch.device('cuda')

    if cfg.mapping_type == 'mlp':
        cfg.prefix_length = cfg.prefix_length or 10
        cfg.clip_model_type = cfg.clip_model_type or 'ViT-B/32'
        cfg.normalize_prefix = cfg.normalize_prefix or False
    elif cfg.mapping_type == 'transformer':
        cfg.prefix_length = cfg.prefix_length or 40
        cfg.clip_model_type = cfg.clip_model_type or 'RN50x4'
        cfg.normalize_prefix = cfg.normalize_prefix or True
    cfg.num_layers = 8

    # Load data and model
    seed = cfg.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    dataset = load_data_vqax(cfg, 'train')
    val_dataset = load_data_vqax(cfg, 'val',eval=True)
    model = load_model(cfg, cfg.pretrained_model)
    model.to(device)
    # Logging & run training/val loops

    if cfg.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    model_without_ddp.load_state_dict(torch.load(cfg.pretrained_model, map_location=torch.device('cpu')))

    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.log_dir, 'checkpoints'), exist_ok=True)

    # with SummaryWriter(log_dir=cfg.log_dir) as writer:
    run(dataset, val_dataset, model, cfg, )#model_ddp=model_without_ddp)

## Saving and loading configs

def save_config(cfg: argparse.Namespace):
    with open(os.path.join(cfg.log_dir, "model_config.json"), 'w') as outfile:
        json.dump({
            'mapping_type' : cfg.mapping_type,
            'prefix_length' : cfg.prefix_length,
            'clip_model_type' : cfg.clip_model_type,
            'normalize_prefix' : cfg.normalize_prefix,
            'num_layers' : cfg.num_layers,
            'prompt_with_choices' : cfg.prompt_with_choices,
            'generation_target' : cfg.generation_target
        }, outfile)

def load_config(config_path: str):
    return argparse.Namespace(
        **json.load(open(config_path))
    )

## Training functions

def run(
    dataset: ClipCocoDataset, val_dataset: ClipCocoDataset, model: ClipCapVQAModel,
    cfg, model_ddp = None
):
    save_config(cfg)

    # accelerator = Accelerator()
    # device = accelerator.device
    device = torch.device('cuda')
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[10,20])
    if cfg.distributed:
        sampler_train = DistributedSampler(dataset)
        sampler_val = DistributedSampler(val_dataset, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset)
        sampler_val = torch.utils.data.SequentialSampler(val_dataset)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, cfg.bs, drop_last=True)
    train_dataloader = DataLoader(dataset, batch_sampler=batch_sampler_train,num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=1, sampler=sampler_val, shuffle=False,num_workers=8)

    for epoch in range(1, cfg.epochs + 1):
        if cfg.distributed:
            sampler_train.set_epoch(epoch)

        model = loop(
            'train', epoch, model, train_dataloader, device,
            cfg.generation_target, cfg.prefix_length, tokenizer=dataset.tokenizer,
            train=True, optimizer=optimizer,

        )

        if epoch % cfg.save_every == 0 or epoch == cfg.epochs - 1:
            if cfg.rank == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(cfg.log_dir, 'checkpoints', f"ckpt-{epoch:03d}.pt"),
                )
        print ('save success')
        # print (optimizer.state_dict()['param_groups'][0]['lr'])
        # scheduler.step()
        with torch.no_grad():
            model,results = loop('val', epoch, model, val_dataloader, device,
                cfg.generation_target, cfg.prefix_length, tokenizer=val_dataset.tokenizer,
                train=False, )
        results['epoch'] = epoch
        output_dir = Path('VQAX/logs')
        if output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(results) + "\n")
        # if epoch % cfg.save_every == 0 or epoch == cfg.epochs - 1:
            # accelerator.wait_for_everyone()
            # unwrapped_model = accelerator.unwrap_model(model)
            #
            # accelerator.save(unwrapped_model.state_dict(),
            #                  os.path.join(cfg.log_dir, 'checkpoints', f"ckpt-{epoch:03d}.pt")
            #                  )
    return model


def loop(
    split, epoch,
    model,dataloader, device,
    generation_target, prefix_len, tokenizer=None,
    train=True, optimizer=None,
    tb_writer=None
):
    assert split in ['train', 'val', 'test', 'test_w_ans']
    print(f">>> epoch {epoch}: {split} set " + ('(train)' if train else '(eval)'))
    sys.stdout.flush()

    if train:
        model.train()
        total_loss = 0.0
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Epoch: [{}]'.format(epoch)
        print_freq = 10
        for context, answer in metric_logger.log_every(dataloader,print_freq,header):
            if train:
                model.zero_grad()
            prefix = context['prefix'].to(device)
            input_tokens_context = context['input'].to(device)
            prompt_len_context = context['len'].to(device)
            target_len_context = context['target_len'].to(device)

            input_tokens_answer = answer['input'].to(device)
            prompt_len_answer = answer['len'].to(device)
            target_len_answer = answer['target_len'].to(device)

            loss_a , answer_logits = compute_step(model, prefix, input_tokens_answer, prefix_len, prompt_len_answer, target_len_answer, None, tokenizer)
            loss_e , _ = compute_step(model, prefix, input_tokens_context, prefix_len, prompt_len_context, target_len_context,None, tokenizer)
            loss_r = reinforcement(model.module, epoch ,prefix, input_tokens_context,input_tokens_answer, prefix_len,
                                   prompt_len_context, prompt_len_answer,target_len_context,target_len_answer, None, tokenizer , baselogist=answer_logits ,device=device)
            loss = loss_a+10*loss_e+loss_r
            if train:
                # loss.backward()
                # accelerator.backward(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()
        metric_logger.synchronize_between_processes()
        return model
    else:
        model.eval()
        metric_logger = utils.MetricLogger(delimiter="  ")
        print('begin eval')
        # gt_answer = {}
        # with open('okvqa/val.json','r') as f:
        #     answer_list = json.load(f)
        # for a in answer_list:
        #     gt_answer[str(a['question_id'])] = a['label']


        answer_list = load_aokvqa('AOKVQA/aokvqa', 'val')
        gt_reation = {}
        gt_answer = {}
        gt_result = {}
        for a in answer_list:
            gt_answer[a['question_id']] = a['direct_answers']
            gt_reation[a['question_id']] = []
            for j in a['rationales']:
                gt_reation[a['question_id']].append({"image_id":a["question_id"], "caption": j})
            gt_result[a['question_id']] = a
        ###########vqax#########
        # with open('vqax/vqaX_test_va.json','r') as f:
        #     answer_list = json.load(f)
        # gt_reation = {}
        # gt_answer = {}
        # gt_result = {}
        # for a in answer_list:
        #     gt_answer[a['question_id']] = a['answers']
        #     gt_reation[a['question_id']] = []
        #     for j in a['rationales']:
        #         gt_reation[a['question_id']].append({"image_id":a["question_id"], "caption": j})
        #     gt_result[a['question_id']] = a

        with torch.no_grad():
            reation_list = []
            answer_list = []
            question_list =[]
            print_freq = 100
            header = 'Test:'


            for a in metric_logger.log_every(dataloader,print_freq,header):
                # q = dataloader.dataset.question_ids[i]
                prefix, prompt_tokens, prompt_len, q = a[0], a[1], a[2], a[3][0]#,a[4]
                prefix = prefix.to(device)
                prompt_tokens = prompt_tokens.to(device)
                embedding_text = model.module.gpt.transformer.wte(prompt_tokens)
                prefix_projections = model.module.clip_project(prefix).view(-1, model.module.prefix_length, model.module.gpt_embedding_size)
                embed = torch.cat((prefix_projections, embedding_text), dim=1)
                # if cfg.beam_search:

                generated_text = generate_beam_val(model.module, tokenizer, embed, device, beam_size=5, return_top_pred=True,
                                               entry_length=87, stop_token_index=tokenizer.eos_token_id)
                print (generated_text)
                reation_list.extend(list(utils.all_gather(generated_text)))

                generated_text = torch.tensor(dataloader.dataset.tokenizer(generated_text + ' ' + 'answer:')['input_ids'],dtype=torch.long).cuda()
                generated_text = model.module.gpt.transformer.wte(generated_text).unsqueeze(0)
                e = torch.cat([embed, generated_text], dim=1)
                answer = generate_beam_val(model.module, tokenizer, e, device, beam_size=3, return_top_pred=True,
                                       entry_length=87, stop_token_index=tokenizer.eos_token_id,answer=True)
                print (answer)
                answer_list.extend(list(utils.all_gather(answer)))
                question_list.extend(list(utils.all_gather(q)))

            metric_logger.synchronize_between_processes()
            gen = {}
            predictions = {}
            for q,r,a, in zip(question_list,reation_list,answer_list):
                gen[q] = []
                gen[q].append({"image_id": q,"caption": r})
                predictions[q] = a
            acc = []
            gen_vqax = {}
            gt_vqax = {}
            right={}
            for q in predictions.keys():
                pred = predictions[q]
                direct_answer = gt_answer[q]
                gt_re = gt_result[q]
                num_match = sum([pred==da for da in direct_answer])
                vqa_acc = min(1.0, num_match/3.0)
                # if pred in direct_answer:
                #     vqa_acc = 1
                # else:
                #     vqa_acc = 0
                # if pred == max(direct_answer,key=direct_answer.count):
                if pred in direct_answer:
                    gen_vqax[q] = gen[q]
                    gt_vqax[q] = gt_reation[q]
                    ####
                    img_path = '/home/B/suowei/coco2017/val2017/'+ (12-len(str(gt_re['image_id'])))*'0'+str(gt_re['image_id'])+'.jpg'
                    img = cv2.imread(img_path)
                    cv2.imwrite('result_aokvqa/img/'+str(q)+'.jpg',img)
                    right[str(q)] = {'ques':gt_re['question'],
                                     'ans':pred,
                                     'pred_exp':gen[q][0]['caption'],

                                     'gt_exp':gt_re['rationales']}
                acc.append(vqa_acc)
            with open('result_aokvqa/vqax_pred.json','w') as f:
                json.dump(right,f)

            acc = sum(acc) / len(acc) *100
            print ('acc:' + str(acc))
            # cocoEval = COCOEvalCap(gt_reation, gen)
            cocoEval = COCOEvalCap(gt_vqax,gen_vqax)
            results = cocoEval.evaluate()
            results['acc'] = acc
        metric_logger.synchronize_between_processes()

            # acc = sum(acc) / len(acc) * 100
            # print('acc:' + str(acc))

            # cocoEval,_ = compute_scores(gtss, gen)
            # val_cider = cocoEval['CIDEr']
            # print ('data/val_cider', val_cider)
            # print ('data/val_bleu1', cocoEval['BLEU'][0])
            # print ('data/val_bleu4', cocoEval['BLEU'][3])
            # print ('data/val_meteor', cocoEval['METEOR'])
            # print ('data/val_rouge', cocoEval['ROUGE'])
        # cocoEval.evaluatecocoEval

    return model,results


def compute_step( model, prefix, input_tokens,
                  prefix_len, prompt_len, target_len,
                  metrics=None, tokenizer=None ):

    outputs = model(prefix, input_tokens)

    ## Compute loss (comparing [target, eos] indices)

    target_logits = [
        l[s:e] for l, s, e in zip(
            outputs.logits,
            prefix_len + prompt_len - 1,
            prefix_len + prompt_len + target_len
        )
    ]

    target_tokens = [
        t[s:e] for t, s, e in zip(
            input_tokens,
            prompt_len,
            prompt_len + target_len + 1
        )
    ]
    predict_list = []
    target_list = []
    for predict, target in zip(target_logits, target_tokens):
        if 3 not in target:
            predict_list.append(predict)
            target_list.append(target)
    if len(predict_list) == 0:
        loss = 0
    else:
        loss =  F.cross_entropy(
            torch.cat(predict_list),
            torch.cat(target_list),
        )
    # loss = F.cross_entropy(
    #     torch.cat(target_logits),
    #     torch.cat(target_tokens),
    #
    # )


    return loss , target_logits

def reinforcement( model,epoch, prefix, input_tokens,input_tokens_answer,
                  prefix_len, prompt_len, prompt_len_answer,target_len,target_len_answer,
                  metrics=None, tokenizer=None , baselogist = None , num_beam = 2 , device = 'cpu'):
    embedding_text = model.gpt.transformer.wte(input_tokens)
    prefix_projections = model.clip_project(prefix).view(-1, model.prefix_length, model.gpt_embedding_size)
    embed = torch.cat((prefix_projections, embedding_text), dim=1)
    loss_r = 0.
    reward_loss = 0.
    text = []
    text_label = []
    start = []

    for j in range(num_beam):
        text_beam = []
        text_label_beam = []
        start_beam = []
        text.append(text_beam)
        text_label.append(text_label_beam)
        start.append(start_beam)
    # eos_token = torch.tensor([tokenizer.eos_token_id], dtype=torch.int64)
    idx = 0
    idxx = 0
    for i in range(prefix.shape[0]):
        # if 3 not in input_tokens[i,:]:
        #     true_embdding = torch.cat([embed[i,:-1,:].unsqueeze(0),model.gpt.transformer.wte(input_tokens_answer[i, prompt_len[i] - 2:].unsqueeze(0))],dim=1)
        #     len = torch.nonzero(input_tokens[i]).shape[0]-1
        #     true_logs = model.gpt(inputs_embeds=true_embdding).logits[0,prefix_len+len-1:prefix_len+len+target_len_answer[i]+2,:]
        #     target = input_tokens_answer[i, prompt_len[i]-2:prompt_len[i]-2 + target_len_answer[i]+2+1]
        #     loss = F.cross_entropy(true_logs,target)
        #     loss_r = loss_r + loss
            # idx = idx+1
        gt_context_len = torch.nonzero(input_tokens[i,:]).shape[0]
        gt_context = tokenizer.decode(input_tokens[i,prompt_len[i]:gt_context_len-1]).strip()
        outputs , scores = generate_beam(model, tokenizer, embed[i,:prompt_len[i]+prefix_len,:].unsqueeze(0), device, beam_size=num_beam)
        # print (outputs[0])
        outputs.append(gt_context)
        output = []
        for sentence in outputs:
            output.append(' '+sentence)
        target_log = []
        target_tokens = []
        for j in range(num_beam+1):
            outputs = torch.tensor(tokenizer(output[j])['input_ids'], dtype=torch.long).to(device)
            # outputs = torch.cat([outputs,eos_token.to(outputs.device)],dim=0)
            pre_e_len = outputs.shape[0] + 2
            wei_embedding=torch.cat([embed[i, :prompt_len[i] + prefix_len, :].unsqueeze(0), model.gpt.transformer.wte(outputs).unsqueeze(0),
                           model.gpt.transformer.wte(input_tokens_answer[i, prompt_len[i] - 2:].unsqueeze(0))], dim=1)
            r_log = model.gpt(inputs_embeds=wei_embedding).logits[0,prefix_len+prompt_len[i] + pre_e_len - 1:prefix_len + prompt_len[i] + pre_e_len + target_len_answer[i],:]
            r_log_soft = F.softmax(r_log,dim=-1)
            target_log.append(r_log)
            b_log_soft = F.softmax(baselogist[i],dim=-1)
            true_answer = input_tokens_answer[i, prompt_len[i]:prompt_len[i] + target_len_answer [i]+ 1]
            target_tokens.append(true_answer)
            gt_score = 0.
            wei_score = 0.
            for gt, id, wei in zip(b_log_soft, true_answer, r_log_soft):
                gt_score += gt[id]
                wei_score += wei[id]
            gt_score = gt_score / r_log.shape[0]
            wei_score = wei_score / r_log.shape[0]
            if j==num_beam:
                if 3 not in input_tokens[i,:]:
                    loss_r = F.cross_entropy(r_log, true_answer) + loss_r
                    idx = idx+1
            else:
                # if 3 in input_tokens[i,:]:
                    reward_loss = torch.clamp((wei_score - gt_score)*10*scores[j],min=0.) + reward_loss
                    idxx = idxx + 1
                    if reward_loss>0:
                        reward_loss = F.cross_entropy(r_log, true_answer) + reward_loss
                        idxx = idxx + 1
            # if j!=num_beam:
            #     if 3 in input_tokens[i,:]:
            #         loss_r = F.cross_entropy(r_log, true_answer) + loss_r
            #         idx = idx + 1



    reward_loss = reward_loss/(idxx+1e-8)
    loss_r = (loss_r / (idx+1e-8))
    # print (reward_loss)

    return reward_loss + loss_r*10

def compute_exact(a_gold, a_pred):
    return int(a_gold == a_pred)

def compute_f1(a_gold, a_pred):
    gold_toks = a_gold.split()
    pred_toks = a_pred.split()
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def generate_beam(model, tokenizer, embed, device, beam_size: int = 5, return_top_pred: bool = False, entry_length=20, temperature=1.0, stop_token_index: int = 50256):
    generated = embed
    tokens = None

    scores = None
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    scores_list = None
    for _ in range(entry_length):
        outputs = model.gpt(inputs_embeds=generated)

        logits = outputs.logits
        logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
        logits = logits.softmax(-1).log()

        if scores is None:
            scores, next_tokens = logits.topk(beam_size, -1)
            generated = generated.expand(beam_size, *generated.shape[1:])
            next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
            scores_list = scores.unsqueeze(-1)  # Each array holds the maximum probability of all words

            if tokens is None:
                tokens = next_tokens
            else:
                tokens = tokens.expand(beam_size, *tokens.shape[1:])
                tokens = torch.cat((tokens, next_tokens), dim=1)
        else:
            logits[is_stopped] = -float(np.inf)
            logits[is_stopped, 0] = 0
            scores_sum = scores[:, None] + logits
            seq_lengths[~is_stopped] += 1
            scores_sum_average = scores_sum / seq_lengths[:, None]
            scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(
                beam_size, -1
            )
            next_tokens_source = next_tokens // scores_sum.shape[1]
            seq_lengths = seq_lengths[next_tokens_source]
            next_tokens = next_tokens % scores_sum.shape[1]
            scores_list = torch.cat(
                [scores_list[next_tokens_source], logits[next_tokens_source, next_tokens].unsqueeze(-1)], dim=-1)
            next_tokens = next_tokens.unsqueeze(1)
            tokens = tokens[next_tokens_source]
            tokens = torch.cat((tokens, next_tokens), dim=1)
            generated = generated[next_tokens_source]
            scores = scores_sum_average * seq_lengths
            is_stopped = is_stopped[next_tokens_source]

        next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
        generated = torch.cat((generated, next_token_embed), dim=1)
        is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
        if is_stopped.all():
            break

    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [
        tokenizer.decode(output[: int(length) - 1]).strip()
        for output, length in zip(output_list, seq_lengths)
    ]
    #token = []
    #token.append(torch.tensor(tokenizer(output_texts[0])['input_ids'], dtype=torch.int64).to(device))
    scores = -torch.mean(scores_list[0, :tokens.shape[1]]).unsqueeze(0)
    for i in range(1, beam_size):
        #token.append(torch.tensor(tokenizer(output_texts[i])['input_ids'], dtype=torch.int64).to(device))
        scores = torch.cat((scores, -torch.mean(scores_list[i, :tokens.shape[1]]).unsqueeze(0)), dim=0)

    return output_texts,scores


def generate_beam_val(model, tokenizer, embed, device, beam_size: int = 5, return_top_pred: bool = False, entry_length=67, temperature=1.0, stop_token_index: int = 50256,answer=False):
    generated = embed
    tokens = None

    scores = None
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)

    for _ in range(entry_length):
        outputs = model.gpt(inputs_embeds=generated)

        logits = outputs.logits
        logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
        logits = logits.softmax(-1).log()

        if scores is None:
            scores, next_tokens = logits.topk(beam_size, -1)
            generated = generated.expand(beam_size, *generated.shape[1:])
            next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
            if tokens is None:
                tokens = next_tokens
            else:
                tokens = tokens.expand(beam_size, *tokens.shape[1:])
                tokens = torch.cat((tokens, next_tokens), dim=1)
        else:
            logits[is_stopped] = -float(np.inf)
            logits[is_stopped, 0] = 0
            scores_sum = scores[:, None] + logits
            seq_lengths[~is_stopped] += 1
            scores_sum_average = scores_sum / seq_lengths[:, None]
            scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(
                beam_size, -1
            )
            next_tokens_source = next_tokens // scores_sum.shape[1]
            seq_lengths = seq_lengths[next_tokens_source]
            next_tokens = next_tokens % scores_sum.shape[1]
            next_tokens = next_tokens.unsqueeze(1)
            tokens = tokens[next_tokens_source]
            tokens = torch.cat((tokens, next_tokens), dim=1)
            generated = generated[next_tokens_source]
            scores = scores_sum_average * seq_lengths
            is_stopped = is_stopped[next_tokens_source]

        next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
        generated = torch.cat((generated, next_token_embed), dim=1)
        is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
        if is_stopped.all():
            break

    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    # eos_token = torch.tensor([tokenizer.eos_token_id], dtype=torch.int64)
    if answer:
        output_texts = [tokenizer.decode(output[: int(length) - 1]).strip() for output, length in zip(output_list, seq_lengths)]
    else:
        output_texts = [tokenizer.decode(output[: int(length)-1 ]).strip() for output, length in zip(output_list, seq_lengths)]

    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]

    if return_top_pred:
        return output_texts[0]
    return output_texts


if __name__ == '__main__':
    main()
