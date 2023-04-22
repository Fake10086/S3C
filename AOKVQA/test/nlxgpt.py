# test in vqax save related data
import sys
import os
sys.path.extend(['/home/B/lws/code/nlx-gpt', ])
sys.path.append('...')
sys.path.append('..')
import random
import torch
from tqdm import tqdm
import torch.utils.data
import torchvision.transforms as transforms
from transformers import GPT2Tokenizer
from transformers import AdamW
from VQAX.test.cococaption.pycocoevalcap.eval import COCOEvalCap
from accelerate import Accelerator
from model.gpt import GPT2LMHeadModel
from model.clip_vit import ImageEncoder
from dataset import  nlxgpt as aok_test
from transformers import (
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    StoppingCriteriaList,
    MaxLengthCriteria,
    BeamSearchScorer,
    GPT2LMHeadModel
)
random.seed(1)
torch.manual_seed(1)
import numpy as np
np.random.seed(1)
torch.backends.cudnn.deterministic = True

def load_pretrained():
    model_path = '../../pretrain_model/pretrain_model'
    tokenizer_path = './../pretrain_model/pretrain_tokenizer_0'
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)  # load tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)  # load model with config
    return tokenizer, model
def change_requires_grad(model, req_grad):
    for p in model.parameters():
        p.requires_grad = req_grad
def get_optimizer(model, learning_rate):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    return optimizer

def load_checkpoint(log_path, epoch=0):
    model_name = 'nle_model_{}'.format(str(epoch))
    tokenizer_name = '../../pretrain_model/pretrain_tokenizer_0'
    filename = 'ckpt_stats_' + str(epoch) + '.tar'

    tokenizer = GPT2Tokenizer.from_pretrained( tokenizer_name)  # load tokenizer
    model = GPT2LMHeadModel.from_pretrained(log_path+model_name).to(device)  # load model with config
    opt = torch.load(log_path + filename)
    optimizer = get_optimizer(model, learning_rate)
    optimizer.load_state_dict(opt['optimizer_state_dict'])
    start_epoch = opt['epoch'] + 1
    scheduler_dic = opt['scheduler']
    del opt
    torch.cuda.empty_cache()

    return tokenizer, model, optimizer, scheduler_dic, start_epoch


# main:
accelerator = Accelerator()
device = accelerator.device

finetune_pretrained = True  # if True, finetunes from the image captioning model
eval_batch_size = 1
img_size = 224
pre_path = '../logs/nlx/'
ckpt_path = '../logs/nlx/'
nle_data_test_path = '../aokvqa/aokvqa_v1p0_val.json'
max_seq_len = 40

no_sample = True
top_k = 0
top_p = 0.9
weight_decay = 0
learning_rate = 2e-5 if not finetune_pretrained else 1e-5
gradient_accumulation_steps = 1
start_epoch = 0
temperature = 1
num_beam = 4
image_encoder = ImageEncoder(device).to(device)
change_requires_grad(image_encoder, False)

for epoch in range(10,11):
    load_from_epoch = epoch
    tokenizer, model, optimizer, scheduler_dic, start_epoch = load_checkpoint(pre_path, load_from_epoch)
    print("Model Setup Ready...")


    img_transform = transforms.Compose([transforms.Resize((img_size, img_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    test_dataset = aok_test.AOKEvalDataset(path=nle_data_test_path,
                                   transform=img_transform,
                                   tokenizer=tokenizer,
                                   max_seq_len=max_seq_len)

    model, optimizer = accelerator.prepare(model, optimizer)
    model.eval()
    result = {}
    target = {}
    answer = {}
    data = test_dataset.data
    for i in data:
        target[i['question_id']] = []
        for j in i['rationales'] :
            target[i['question_id']].append({
                'image_id':i['image_id'],
                'caption':j,
            })
    for i in data:
        answer[i['question_id']] = i['direct_answers']
    acc = []
    test_qa = False
    correct = 0
    with torch.no_grad():
        if test_qa:
            for i in tqdm(range(len(test_dataset))):
                img, qid, _, input_ids, sigment = test_dataset[i]
                img = img.to(device).unsqueeze(0)

                input_ids = input_ids.to(device)
                img_embeddings = image_encoder(img)
                qid = str(qid.item())
                model_kwargs = {
                    "encoder_hidden_states": img_embeddings,
                }
                beam_scorer_context = BeamSearchScorer(
                    batch_size=1,
                    num_beams=num_beam,
                    device=device,
                    do_early_stopping=True,

                )
                stopping_criteria_context = StoppingCriteriaList([MaxLengthCriteria(max_length=40)])
                logits_processor_context = LogitsProcessorList(
                    [
                        MinLengthLogitsProcessor(3, eos_token_id=model.config.eos_token_id),
                    ]
                )
                ones = torch.ones((num_beam, 1), dtype=torch.long).to(device)
                input_ids_true = ones * input_ids
                outputs = model.beam_search(input_ids_true, beam_scorer_context,
                                            logits_processor=logits_processor_context, output_scores=True,
                                            stopping_criteria=stopping_criteria_context, pad_token_id=50257,
                                            **model_kwargs)
                outputs = outputs[0]
                # get sentence
                pre_ans = tokenizer.decode(outputs[len(input_ids):], skip_special_tokens=True)
                pre_ans = pre_ans[1:]  # ' '
                print('input_ids:' + tokenizer.batch_decode(input_ids)[0])
                print('pre_answer:' + pre_ans)
                tar_ans = data[qid]['answers']
                tar_ans_list = []
                for ans in tar_ans:
                    tar_ans_list.append(ans['answer'])
                if pre_ans in tar_ans_list:
                    correct = correct + 1
                print('acc:' + str(correct / len(test_dataset)))
        else:
            print("epoch: "+str(epoch)+" start!")
            for i in range(len(test_dataset)):
                img, qid, input_ids = test_dataset[i]
                img = img.to(device).unsqueeze(0)

                input_ids = input_ids.to(device)
                img_embeddings = image_encoder(img)

                model_kwargs = {
                    "encoder_hidden_states": img_embeddings
                }
                beam_scorer_context = BeamSearchScorer(
                    batch_size=1,
                    num_beams=num_beam,
                    device=device,
                    do_early_stopping=True,

                )
                stopping_criteria_context = StoppingCriteriaList([MaxLengthCriteria(max_length=40)])
                logits_processor_context = LogitsProcessorList(
                    [
                        MinLengthLogitsProcessor(3, eos_token_id=model.config.eos_token_id),
                    ]
                )
                ones = torch.ones((num_beam, 1), dtype=torch.long).to(device)
                input_ids_true = ones * input_ids
                outputs = model.beam_search(input_ids_true, beam_scorer_context,
                                            logits_processor=logits_processor_context, output_scores=True,
                                            stopping_criteria=stopping_criteria_context, pad_token_id=50257,
                                            **model_kwargs)
                outputs = outputs[0]
                # get sentence
                caption = tokenizer.decode(outputs, skip_special_tokens=True)
                explanation = outputs[len(input_ids):]
                explanation = tokenizer.decode(explanation, skip_special_tokens=True)
                result[qid] = []
                result[qid].append({
                    'image_id': data[i]['image_id'],
                    'caption': explanation[1:]
                })

                prompt = ' answer:'
                prompt = torch.tensor(tokenizer(prompt)['input_ids'], dtype=torch.long).to(device)
                input_ids = torch.cat(
                    (torch.tensor(tokenizer(caption)['input_ids'], dtype=torch.long).to(device), prompt), dim=0)

                # generate answer
                model_kwargs = {
                    "encoder_hidden_states": img_embeddings
                }
                beam_scorer_context = BeamSearchScorer(
                    batch_size=1,
                    num_beams=num_beam,
                    device=device,
                    do_early_stopping=True,

                )
                stopping_criteria_context = StoppingCriteriaList([MaxLengthCriteria(max_length=45)])
                logits_processor_context = LogitsProcessorList(
                    [
                        MinLengthLogitsProcessor(3, eos_token_id=model.config.eos_token_id),
                    ]
                )
                ones = torch.ones((num_beam, 1), dtype=torch.long).to(device)
                input_ids_true = ones * input_ids
                outputs = model.beam_search(input_ids_true, beam_scorer_context,
                                            logits_processor=logits_processor_context, output_scores=True,
                                            stopping_criteria=stopping_criteria_context, pad_token_id=50257,
                                            **model_kwargs)

                pre_ans = outputs[0][len(input_ids):]
                pre_ans = tokenizer.decode(pre_ans, skip_special_tokens=True)
                pre_ans = pre_ans[1:]  # ' '
                answer_list = answer[qid]
                num_match = sum([pre_ans == da for da in answer_list])
                vqa_acc = min(1.0, num_match / 3.0)
                acc.append(vqa_acc)
                val_acc = sum(acc) / len(acc) * 100
                if i % 100 == 0 or i == (len(test_dataset) - 1):
                    print('acc:' + str(val_acc) + f"{i} / {len(test_dataset)}")

    acc = sum(acc) / len(acc) * 100
    print('acc:' + str(acc))
    cocoEval = COCOEvalCap(target, result)
    cocoEval.evaluate()





