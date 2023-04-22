from typing import Optional
import argparse
import pathlib
import pickle
import random
import  json
import tqdm
import torch
from transformers import GPT2Tokenizer
import sys
sys.path.append("...")
from utils.load_data import load_aokvqa
from model.train_clipcap import  ClipCocoDataset


def load_data(cfg: argparse.Namespace, split: str, eval: bool = False):
    features = vars(cfg).get(f"{split}_features", None)#.name
    return AokvqaDataset(
        cfg.aokvqa_dir, split, features,
        cfg.prompt_with_choices, cfg.generation_target,
        cfg.prefix_length, normalize_prefix=cfg.normalize_prefix,
        gpt2_type='gpt2',
        eval=eval
    )

def load_data_vqax(cfg: argparse.Namespace, split: str, eval: bool = False):
    # features = vars(cfg).get(f"{split}_features", None)#.name
    if eval:
        features = torch.load('features/vqa_val.pt')
    else:
        features = torch.load('features/vqa_train.pt')
    return AokvqaDataset_vqax(
        cfg.aokvqa_dir, split, features,
        cfg.prompt_with_choices, cfg.generation_target,
        cfg.prefix_length, normalize_prefix=cfg.normalize_prefix,
        gpt2_type='gpt2',
        eval=eval
    )

class AokvqaDataset_vqax(ClipCocoDataset):
    def __init__(self,
        aokvqa_dir: pathlib.Path, split: str, features: Optional[pathlib.Path],
        prompt_with_choices: bool, generation_target: str,
        prefix_length: int, normalize_prefix: bool, gpt2_type: str,
        eval: bool = False
    ):
        self.split = split
        self.grunding_true = []
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        self.eval = eval
        self.answer_list = []
        # aokvqa_set = load_aokvqa(aokvqa_dir, split)
        if eval:
            with open('vqax/vqaX_test_va.json','r') as f:
                aokvqa_set = json.load(f)
        else:
            with open('vqax/vqaX_train_va.json','r') as f:
                aokvqa_set = json.load(f)
        if features is not None:
            embeddings = features
            self.prefixes = []
        # with open('vqa/train.json') as f:
        #     vqa_train = json.load(f)
        self.question_ids = []
        self.prompt_tokens = []
        self.question_context_answer = []
        self.question_answer = []
        self.context = []
        self.target_tokens_context = []
        self.target_tokens_answer = []
        self.seq_lengths_context = []
        self.seq_lengths_answer = []
        if self.eval is False:
            self.target_tokens_context = []
            self.target_tokens_answer = []
            self.seq_lengths_context = []
            self.seq_lengths_answer = []
        answer_list = []
        qqq = []
        # for i in range(len(aokvqa_set)):
        for i in range(the_number_of_vqa):
            answer_list.append(aokvqa_set[i]['answers'][0])
            d = aokvqa_set[i]
            if ' '.join(d['question'].split()[:4]) not in qqq:
                qqq.append(' '.join(d['question'].split()[:4]))
            q = d['question_id']

            if features is not None:
                self.prefixes.append(embeddings[d['image_id'].split('.')[0]])
            if split == 'val':
                self.answer_list.append(d['answers'])
                self.context.append(d['rationales'])

            self.question_ids.append( q )

            self.prompt_tokens.append(
                torch.tensor(
                    self.tokenizer(
                        prompt_text('rationale', d,include_choices=prompt_with_choices)
                    )['input_ids'], dtype=torch.int64
                )
            )
            q_c_a_list = []
            if split == 'train':
                for i in d['rationales']:
                    q_c_a = prompt_text('rationale', d,include_choices=prompt_with_choices)+' '+i
                    # q_c_a = prompt_text('answer', d, include_choices=prompt_with_choices)
                    q_c_a_list.append(torch.tensor(self.tokenizer(q_c_a)['input_ids'],dtype=torch.int64))
                self.question_context_answer.append(q_c_a_list)

            q_a = prompt_text('answer',d,include_choices=prompt_with_choices)
            self.question_answer.append(torch.tensor(self.tokenizer(q_a)['input_ids'],dtype=torch.int64))

            if self.eval is False:
                self.target_tokens_context.append([
                    torch.tensor(
                        self.tokenizer(t)['input_ids'],
                        dtype=torch.int64
                    ) for t in target_texts(generation_target, d)
                ])

                # self.target_tokens_answer.append([
                #     torch.tensor(
                #         self.tokenizer(t)['input_ids'],
                #         dtype=torch.int64
                #     ) for t in target_texts('answer', d)
                # ])

                self.target_tokens_answer.append([
                    torch.tensor(
                        self.tokenizer(t)['input_ids'],
                        dtype=torch.int64
                    ) for t in d['answers']
                ])

                self.seq_lengths_context += [
                    self.prompt_tokens[-1].shape[0] + t.shape[0]
                    for t in self.target_tokens_context[-1]
                ]
                self.seq_lengths_answer += [
                    self.prompt_tokens[-1].shape[0] + t.shape[0]
                    for t in self.target_tokens_answer[-1]
                ]


        # if split =='train':
        #     # vqa_img_feature = torch.load('features/vqa_train.pt')
        #     for d in tqdm.tqdm(vqa_train[:200000]):
        #         if ' '.join(d['sent'].split()[:4]) not in qqq:
        #             continue
        #         value_raw = 0
        #         for key, value in d['label'].items():
        #             if value > value_raw:
        #                 key_label = key
        #                 value_raw = value
        #         if key not in answer_list:
        #             continue
        #         # if d['answer_type'] != 'other':
        #         #     continue
        #         # if 'color' in d['question_type']:
        #         #     continue
        #         # if str(d['question_id']) in self.question_ids:
        #         #     continue
        #         #     print ('asfd')
        #         # if d['answer_type'] == 'yes/no' or d['answer_type']== 'number':
        #         #     continue
        #         # if  'color' in d['question_type'] or 'what is the' in d['question_type'] or 'what is this' == d['question_type']\
        #         #         or 'what is on the' in d['question_type'] or 'is it' in d['question_type']:
        #         #     continue
        #         # if 'why' in d['question_type']:
        #         self.question_ids.append(str(d['question_id']))
        #         self.prefixes.append(embeddings[d['img_id']])
        #         self.prompt_tokens.append(
        #             torch.tensor(
        #                 self.tokenizer(
        #                     prompt_text(generation_target, d, include_choices=prompt_with_choices, vqa=True)
        #                 )['input_ids'], dtype=torch.int64
        #             )
        #         )
        #         q_c = prompt_text(generation_target, d, include_choices=prompt_with_choices, vqa=True)
        #         q_c = torch.tensor(self.tokenizer(q_c)['input_ids'], dtype=torch.int64)
        #         fu_1 = torch.ones([1], dtype=torch.int64) * 3
        #         q_c = torch.cat([q_c, fu_1], dim=0)
        #         self.question_context_answer.append([q_c])
        #         q_a = prompt_text('answer', d, include_choices=prompt_with_choices, vqa=True)
        #         self.question_answer.append(torch.tensor(self.tokenizer(q_a)['input_ids'], dtype=torch.int64))
        #         # if self.eval is False:
        #         self.target_tokens_context.append([fu_1])
        #         value_raw = 0
        #         for key, value in d['label'].items():
        #             if value > value_raw:
        #                 key_label = key
        #                 value_raw = value
        #         self.target_tokens_answer.append(
        #             [torch.tensor(self.tokenizer(key_label)['input_ids'], dtype=torch.int64)])
        #         # self.target_tokens_answer.append([torch.tensor(self.tokenizer(key)['input_ids'], dtype=torch.int64) for key,value in d['label'].items()])
        #
        #         # self.target_tokens_answer.append(
        #         #     [torch.tensor(self.tokenizer(d['label'][0])['input_ids'], dtype=torch.int64)])
        #         self.seq_lengths_context += [self.prompt_tokens[-1].shape[0] + 1]
        #         self.seq_lengths_answer += [
        #             self.prompt_tokens[-1].shape[0] + self.target_tokens_answer[-1][-1].shape[0]+1]

        if self.eval is False:
            self.max_seq_len_context = max(self.seq_lengths_context)
            self.max_seq_len_answer = max(self.seq_lengths_answer)+1

    def __getitem__(self, i: int):
        prefix = self.prefixes[i]
        if self.normalize_prefix:
            prefix = prefix / prefix.norm(2, -1)
        question_id = self.question_ids[i]
        prompt = self.prompt_tokens[i]# static format 'question:.... context:'
        prompt_len = prompt.shape[0]

        prompt_q_a = self.question_answer[i]
        prompt_q_a_len = prompt_q_a.shape[0]

        # if self.split == 'val':
        #     answer_list = self.answer_list[i]
        if self.eval:
            if self.split =='val':
               # context = self.context[i]
                return prefix, prompt, prompt_len, question_id, #context
            else:
                context = self.context[i]
                return prefix, prompt, prompt_len, question_id, context


        k = random.sample(self.target_tokens_context[i], 1)
        target_context = k[0]
        target_len_context = target_context.shape[0]


        # target_answer = self.target_tokens_answer[i][0]
        k = random.sample(self.target_tokens_answer[i],1)
        target_answer = k[0]
        target_answer_len = target_answer.shape[0]

        eos_token = torch.tensor([self.tokenizer.eos_token_id], dtype=torch.int64)

        padding_context = self.max_seq_len_context - (prompt.shape[0] + target_context.shape[0])
        padding_context = torch.zeros(padding_context, dtype=torch.int64)

        padding_answer = self.max_seq_len_answer - (prompt_q_a.shape[0] + target_answer.shape[0])
        padding_answer = torch.zeros(padding_answer, dtype=torch.int64)


        input_tokens_context = torch.cat([prompt, target_context, eos_token, padding_context])#question+'context'+context+eos+padding
        input_tokens_answer = torch.cat([prompt_q_a, target_answer, eos_token, padding_answer])#question+'context'+context+eos+padding

        # a = self.tokenizer.decode(input_tokens_context)
        # c = self.tokenizer.decode(input_tokens_answer)


        # prompt_q_e_a = torch.cat([prompt, target_context , ], dim=0)
        # prompt_q_e_a_len = prompt_q_e_a.shape[0]

        context = {'prefix':prefix,'input':input_tokens_context,'len':prompt_len,'target_len':target_len_context}
        answer = {'prefix':prefix,'input':input_tokens_answer,'len':prompt_q_a_len,'target_len':target_answer_len}
        return context,answer#,self.grunding_true[i][0]
        #return prefix, input_tokens_context, prompt_len, target_len_context

    def __len__(self) -> int:
        return len(self.question_ids)


class AokvqaDataset(ClipCocoDataset):
    def __init__(self,
        aokvqa_dir: pathlib.Path, split: str, features: Optional[pathlib.Path],
        prompt_with_choices: bool, generation_target: str,
        prefix_length: int, normalize_prefix: bool, gpt2_type: str,
        eval: bool = False
    ):
        self.split = split
        self.grunding_true = []
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        self.eval = eval
        self.answer_list = []
        aokvqa_set = load_aokvqa(aokvqa_dir, split)
        if features is not None:
            embeddings = torch.load(features)
            self.prefixes = []
        with open('okvqa/train.json') as f:
            vqa_train = json.load(f)
        self.question_ids = []
        self.prompt_tokens = []
        self.question_context_answer = []
        self.question_answer = []
        self.context = []
        self.target_tokens_context = []
        self.target_tokens_answer = []
        self.seq_lengths_context = []
        self.seq_lengths_answer = []
        if self.eval is False:
            self.target_tokens_context = []
            self.target_tokens_answer = []
            self.seq_lengths_context = []
            self.seq_lengths_answer = []

        for i in range(len(aokvqa_set)):
            # if split == 'val':
            #     if i > 16:
            #         break
            d = aokvqa_set[i]
            q = d['question_id']

            if features is not None:
                self.prefixes.append(embeddings[q]['image'] )
            if split == 'val':
                self.answer_list.append(d['direct_answers'])
                self.context.append(d['rationales'])

            self.question_ids.append( q )

            self.prompt_tokens.append(
                torch.tensor(
                    self.tokenizer(
                        prompt_text('rationale', d,include_choices=prompt_with_choices)
                    )['input_ids'], dtype=torch.int64
                )
            )
            q_c_a_list = []
            if split == 'train':
                for i in d['rationales']:
                    q_c_a = prompt_text('rationale', d,include_choices=prompt_with_choices)+' '+i
                    # q_c_a = prompt_text('answer', d, include_choices=prompt_with_choices)
                    q_c_a_list.append(torch.tensor(self.tokenizer(q_c_a)['input_ids'],dtype=torch.int64))
                self.question_context_answer.append(q_c_a_list)

            q_a = prompt_text('answer',d,include_choices=prompt_with_choices)
            self.question_answer.append(torch.tensor(self.tokenizer(q_a)['input_ids'],dtype=torch.int64))

            if self.eval is False:
                self.target_tokens_context.append([
                    torch.tensor(
                        self.tokenizer(t)['input_ids'],
                        dtype=torch.int64
                    ) for t in target_texts(generation_target, d)
                ])

                # self.target_tokens_answer.append([
                #     torch.tensor(
                #         self.tokenizer(t)['input_ids'],
                #         dtype=torch.int64
                #     ) for t in target_texts('answer', d)
                # ])

                self.target_tokens_answer.append([
                    torch.tensor(
                        self.tokenizer(t)['input_ids'],
                        dtype=torch.int64
                    ) for t in d['direct_answers']
                ])

                self.seq_lengths_context += [
                    self.prompt_tokens[-1].shape[0] + t.shape[0]
                    for t in self.target_tokens_context[-1]
                ]
                self.seq_lengths_answer += [
                    self.prompt_tokens[-1].shape[0] + t.shape[0]
                    for t in self.target_tokens_answer[-1]
                ]

        if split =='train':
            vqa_img_feature = torch.load('features/vqa_train.pt')
            for d in tqdm.tqdm(vqa_train):
                # if d['answer_type'] == 'yes/no' or d['answer_type']== 'number':
                #     continue
                # if  'color' in d['question_type'] or 'what is the' in d['question_type'] or 'what is this' == d['question_type']\
                #         or 'what is on the' in d['question_type'] or 'is it' in d['question_type']:
                #     continue
                # if 'why' in d['question_type']:
                self.question_ids.append(str(d['question_id']))
                self.prefixes.append(vqa_img_feature[d['img_id']])
                self.prompt_tokens.append(
                    torch.tensor(
                        self.tokenizer(
                            prompt_text(generation_target, d, include_choices=prompt_with_choices, vqa=True)
                        )['input_ids'], dtype=torch.int64
                    )
                )
                q_c = prompt_text(generation_target, d, include_choices=prompt_with_choices, vqa=True)
                q_c = torch.tensor(self.tokenizer(q_c)['input_ids'], dtype=torch.int64)
                fu_1 = torch.ones([1], dtype=torch.int64) * 3
                q_c = torch.cat([q_c, fu_1], dim=0)
                self.question_context_answer.append([q_c])
                q_a = prompt_text('answer', d, include_choices=prompt_with_choices, vqa=True)
                self.question_answer.append(torch.tensor(self.tokenizer(q_a)['input_ids'], dtype=torch.int64))
                # if self.eval is False:
                self.target_tokens_context.append([fu_1])
                # value_raw = 0
                # for key, value in d['label'].items():
                #     if value > value_raw:
                #         key_label = key
                #         value_raw = value
                # self.target_tokens_answer.append(
                #     [torch.tensor(self.tokenizer(key_label)['input_ids'], dtype=torch.int64)])
                self.target_tokens_answer.append([torch.tensor(self.tokenizer(key_label)['input_ids'], dtype=torch.int64) for key_label in d['label']])
                # self.target_tokens_answer.append(
                #     [torch.tensor(self.tokenizer(d['label'][0])['input_ids'], dtype=torch.int64)])
                self.seq_lengths_context += [self.prompt_tokens[-1].shape[0] + 1]
                self.seq_lengths_answer += [
                    self.prompt_tokens[-1].shape[0] + self.target_tokens_answer[-1][-1].shape[0]]
        if self.eval is False:
            self.max_seq_len_context = max(self.seq_lengths_context)
            self.max_seq_len_answer = max(self.seq_lengths_answer)

    def __getitem__(self, i: int):
        prefix = self.prefixes[i]
        if self.normalize_prefix:
            prefix = prefix / prefix.norm(2, -1)
        question_id = self.question_ids[i]
        prompt = self.prompt_tokens[i]# static format 'question:.... context:'
        prompt_len = prompt.shape[0]

        prompt_q_a = self.question_answer[i]
        prompt_q_a_len = prompt_q_a.shape[0]

        # if self.split == 'val':
        #     answer_list = self.answer_list[i]
        if self.eval:
            if self.split =='val':
               # context = self.context[i]
                return prefix, prompt, prompt_len, question_id, #context
            else:
                context = self.context[i]
                return prefix, prompt, prompt_len, question_id, context


        k = random.sample(self.target_tokens_context[i], 1)
        target_context = k[0]
        target_len_context = target_context.shape[0]


        # target_answer = self.target_tokens_answer[i][0]
        k = random.sample(self.target_tokens_answer[i],1)
        target_answer = k[0]
        target_answer_len = target_answer.shape[0]

        eos_token = torch.tensor([self.tokenizer.eos_token_id], dtype=torch.int64)

        padding_context = self.max_seq_len_context - (prompt.shape[0] + target_context.shape[0])
        padding_context = torch.zeros(padding_context, dtype=torch.int64)

        padding_answer = self.max_seq_len_answer - (prompt_q_a.shape[0] + target_answer.shape[0])
        padding_answer = torch.zeros(padding_answer, dtype=torch.int64)

        input_tokens_context = torch.cat([prompt, target_context, eos_token, padding_context])#question+'context'+context+eos+padding
        input_tokens_answer = torch.cat([prompt_q_a, target_answer, eos_token, padding_answer])#question+'context'+context+eos+padding

        # a = self.tokenizer.decode(input_tokens_context)
        # c = self.tokenizer.decode(input_tokens_answer)


        # prompt_q_e_a = torch.cat([prompt, target_context , ], dim=0)
        # prompt_q_e_a_len = prompt_q_e_a.shape[0]

        context = {'prefix':prefix,'input':input_tokens_context,'len':prompt_len,'target_len':target_len_context}
        answer = {'prefix':prefix,'input':input_tokens_answer,'len':prompt_q_a_len,'target_len':target_answer_len}
        return context,answer#,self.grunding_true[i][0]
        #return prefix, input_tokens_context, prompt_len, target_len_context

    def __len__(self) -> int:
        return len(self.question_ids)

#
# def prompt_text(generation_target, d,include_choices=False):
#     return f"question: {d['question']} " + \
#                (f"choices: {', '.join(d['choices'])}. " if include_choices else '') + {'answer': 'answer:', 'rationale' : 'context:'}[generation_target]
def prompt_text(generation_target, d, include_choices=False,vqa=False):
    if vqa:
        return f"question: {d['sent']}" +' '+ {'answer' : 'answer:', 'rationale' : 'context:'}[generation_target]
    else:
        return f"question: {d['question']} " + \
           (f"choices: {', '.join(d['choices'])}. " if include_choices else '') + {'answer' : 'answer:', 'rationale' : 'context:'}[generation_target]
def target_texts(generation_target, d):
    if generation_target == 'answer':
        targets = [d['choices'][d['correct_choice_idx']]]
    elif generation_target == 'rationale':
        targets = d['rationales']
    return [f" {t}" for t in targets]
