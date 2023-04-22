import sys
import os
sys.path.extend(['/home/B/lws/code/python/nlx-gpt', ])
sys.path.append("..")
sys.path.append("...")
from utils.trainer import trainer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from transformers import GPT2Tokenizer, AutoConfig
from transformers import AdamW, get_linear_schedule_with_warmup
import json
from PIL import Image
from accelerate import Accelerator
from model.gpt import GPT2LMHeadModel
from model.clip_vit import ImageEncoder
from dataset import nlxgpt
def change_requires_grad(model, req_grad):
    for p in model.parameters():
        p.requires_grad = req_grad


def load_checkpoint(ckpt_path, epoch):
    model_name = 'nle_model_{}'.format(str(epoch))
    tokenizer_name = 'nle_gpt2_tokenizer_0'
    filename = 'ckpt_stats_' + str(epoch) + '.tar'

    tokenizer = GPT2Tokenizer.from_pretrained(ckpt_path + tokenizer_name)  # load tokenizer
    model = GPT2LMHeadModel.from_pretrained(ckpt_path + model_name).to(device)  # load model with config
    opt = torch.load(ckpt_path + filename)
    optimizer = get_optimizer(model, learning_rate)
    optimizer.load_state_dict(opt['optimizer_state_dict'])
    start_epoch = opt['epoch'] + 1
    scheduler_dic = opt['scheduler']
    del opt
    torch.cuda.empty_cache()

    return tokenizer, model, optimizer, scheduler_dic, start_epoch


def load_pretrained():
    files = os.getcwd()
    files = os.path.dirname(files)
    files = os.path.dirname(files)
    model_path = os.path.join(files,'pretrain_model/pretrain_model')
    tokenizer_path = os.path.join(files,'pretrain_model/pretrain_tokenizer_0')
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)  # load tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)  # load model with config
    return tokenizer, model


def save_checkpoint(epoch, unwrapped_model, optimizer, tokenizer, scheduler, ckpt_path, **kwargs):
    model_name = 'nle_model_{}'.format(str(epoch))
    tokenizer_name = 'nle_gpt2_tokenizer_{}'.format(str(epoch))
    filename = 'ckpt_stats_' + str(epoch) + '.tar'

    if epoch == 0:
        tokenizer.save_pretrained(ckpt_path + tokenizer_name)  # save tokenizer

    unwrapped_model.save_pretrained(ckpt_path + model_name, save_function=accelerator.save)

    opt = {'epoch': epoch,
           'optimizer_state_dict': optimizer.state_dict(),
           'scheduler': scheduler.state_dict(),
           **kwargs}

    accelerator.save(opt, ckpt_path + filename)








def get_optimizer(model, learning_rate):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    return optimizer


accelerator = Accelerator()
device = accelerator.device

finetune_pretrained = True  # if True, finetunes from the image captioning model
eval_batch_size = 1
img_size = 224
ckpt_path = '../logs/nlx/'
load_path = '../logs/nlx/'
nle_data_train_path = '../aokvqa/aokvqa_v1p0_train.json'
max_seq_len = 40
load_from_epoch = None
no_sample = True
top_k = 0
top_p = 0.9
batch_size = 8   # per GPU
num_train_epochs = 30
weight_decay = 0
learning_rate = 2e-5 if not finetune_pretrained else 1e-5
gradient_accumulation_steps = 1
start_epoch = 0
temperature = 1

image_encoder = ImageEncoder(device).to(device)
change_requires_grad(image_encoder, False)

if load_from_epoch is not None:
    tokenizer, model, optimizer, scheduler_dic, start_epoch = load_checkpoint(load_path, load_from_epoch)

else:

    if finetune_pretrained:
        tokenizer, model = load_pretrained()
        optimizer = get_optimizer(model, learning_rate)
    else:
        tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
        orig_num_tokens = len(tokenizer.encoder)

        num_new_tokens = tokenizer.add_special_tokens({'pad_token': '<pad>',
                                                       'additional_special_tokens': ['<question>', '<answer>',
                                                                                     '<explanation>']})

        assert len(tokenizer) == orig_num_tokens + num_new_tokens
        config = AutoConfig.from_pretrained('distilgpt2')

        # Add configs
        setattr(config, 'img_size', None)
        setattr(config, 'max_seq_len', None)
        config.img_size = img_size
        config.max_seq_len = max_seq_len
        config.add_cross_attention = True

        model = GPT2LMHeadModel.from_pretrained('distilgpt2', config=config)
        model.resize_token_embeddings(len(tokenizer))
        model = model.to(device)
        optimizer = get_optimizer(model, learning_rate)

print("Model Setup Ready...")

img_transform = transforms.Compose([transforms.Resize((img_size, img_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train_dataset_true = nlxgpt.AOKVQATrainDataset(path=nle_data_train_path,
                               transform=img_transform,
                               tokenizer=tokenizer,
                               max_seq_len=max_seq_len,
                               )



train_loader = torch.utils.data.DataLoader(train_dataset_true,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           pin_memory=True)




model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

t_total = (len(train_loader) // gradient_accumulation_steps) * num_train_epochs
warmup_steps = 0  # 0.10 * t_total
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

if load_from_epoch is not None:
    scheduler.load_state_dict(scheduler_dic)

for epoch in range(start_epoch, num_train_epochs):

    model.train()
    accum_loss = 0

    for step, batch in enumerate(train_loader):

        qae_len , img,q_a, q_e1,q_e2,q_e3,q_e1_a,q_e2_a,q_e3_a, prompt_q , ans= batch
        q_a_ids = q_a['input_ids'].to(device)
        q_a_len=q_a['len']
        label_q_a = q_a['label'].to(device)
        #1
        q_e1_ids = q_e1['input_ids'].to(device)
        q_e1_len = q_e1['len']
        label_q_e1 = q_e1['label'].to(device)

        q_e1_a_ids = q_e1_a['input_ids'].to(device)
        q_e1_a_len = q_e1_a['len']
        label_q_e1_a = q_e1_a['label'].to(device)
        #2
        q_e2_ids = q_e2['input_ids'].to(device)
        q_e2_len = q_e2['len']
        label_q_e2 = q_e2['label'].to(device)

        q_e2_a_ids = q_e2_a['input_ids'].to(device)
        q_e2_a_len = q_e2_a['len']
        label_q_e2_a = q_e2_a['label'].to(device)
        #3
        q_e3_ids = q_e3['input_ids'].to(device)
        q_e3_len = q_e3['len']
        label_q_e3 = q_e3['label'].to(device)

        q_e3_a_ids = q_e3_a['input_ids'].to(device)
        q_e3_a_len = q_e3_a['len']
        label_q_e3_a = q_e3_a['label'].to(device)

        img =img.to(device)
        img_embeddings = image_encoder(img)

        q_e_ids = torch.cat((q_e1_ids,q_e2_ids,q_e3_ids),dim=0)
        q_e_a_ids = torch.cat((q_e1_a_ids,q_e2_a_ids,q_e3_a_ids),dim=0)
        label_q_e = torch.cat((label_q_e1,label_q_e2,label_q_e3),dim=0)
        label_q_e_a = torch.cat((label_q_e1_a,label_q_e2_a,label_q_e3_a),dim=0)
        # for q
        text = tokenizer(prompt_q)
        text_ans = tokenizer(ans)['input_ids']
        loss = trainer(decoder=model,
                                   prefix=img_embeddings,
                                   q_a_ids=q_a_ids,
                                   q_e_ids=q_e_ids,
                                   q_e_a_ids = q_e_a_ids,
                                   q_a_len=q_a_len,
                                   device=device,
                                   tokenizer=tokenizer,
                                   label_q_a=label_q_a,
                                   label_q_e_a=label_q_e_a,
                                   label_q_e=label_q_e,
                                   text_qc = text,
                                   text_ans=text_ans
                                   )



        loss = loss / gradient_accumulation_steps
        accelerator.backward(loss)
        accum_loss += loss.item()

        if step % gradient_accumulation_steps == 0 or step == len(train_loader) - 1:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            accelerator.print("\rEpoch {} / {}, Iter {} / {}, Loss: {:.3f}".format(epoch,
                                                                                   num_train_epochs,
                                                                                   step, len(train_loader),
                                                                                   accum_loss),
                              end='          ')

            accum_loss = 0

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    save_checkpoint(epoch, unwrapped_model, optimizer, tokenizer, scheduler, ckpt_path)




