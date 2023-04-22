import torch
import json
from torch.utils.data import Dataset
from PIL import Image
import random
import os
import sys
sys.path.append("...")
from utils import data_utils
#sys.path.append("/home/B/lws/code/python/IRS")
# import


class AOKVQATrainDataset(Dataset):

    def __init__(self, path, transform, tokenizer, max_seq_len):

        self.tokenizer = tokenizer
        self.transform = transform
        self.max_seq_len = max_seq_len
        self.data = json.load(open(path, 'r'))
        self.ids_list = len(self.data)

        self.index_tracker = {i: len(self.data[i]["rationales"]) - 1 for i in range(len(self.data))}

    def __getitem__(self, i):

        sample = self.data[i]
        img_id = sample['image_id']

        text_a = data_utils.proc_ques(sample['question'])  # question
        answer = data_utils.proc_ans_aok(sample['direct_answers'])

        text_b = sample['rationales']  # explanation
        ids = random.randint(0,2)
        text_b_1 = text_b[ids]


        q_segment_id, a_segment_id, e_segment_id = self.tokenizer.convert_tokens_to_ids(['<question>',
                                                                                         '<answer>',
                                                                                         '<explanation>'])
        prompt_text_a = ' question: ' + text_a
        prompt_answer = ' answer: ' + answer
        prompt_text_b_1 = ' context: ' + text_b_1

        prompt_q = prompt_text_a + ' context:'

        tokens = self.tokenizer.tokenize(prompt_text_a)
        answer = self.tokenizer.tokenize(prompt_answer)
        tokens_b_1 = self.tokenizer.tokenize(prompt_text_b_1)



        # new
        question_ids = tokens
        answer_ids = answer


        q_len = len(question_ids)
        a_len = len(answer_ids)
        e1_len = len(tokens_b_1)


        # get q + a
        q_a_ids = question_ids + answer_ids + [self.tokenizer.eos_token_id]
        q_a_len = len(q_a_ids)
        pad_len = self.max_seq_len - q_a_len + 1
        q_a_ids = q_a_ids + ([self.tokenizer.pad_token] * (pad_len - 1))

        # get q + e
        q_e1_ids = question_ids + tokens_b_1 + [self.tokenizer.eos_token_id]
        q_e1_len = len(q_e1_ids)
        pad_len = self.max_seq_len - q_e1_len + 1
        q_e1_ids = q_e1_ids + ([self.tokenizer.pad_token] * (pad_len - 1))

        # get q + e + a
        q_e1_a_ids = question_ids + tokens_b_1 + answer_ids + [self.tokenizer.eos_token_id]
        q_e1_a_len = len(q_e1_a_ids)
        pad_len = self.max_seq_len - q_e1_a_len + 1
        q_e1_a_ids = q_e1_a_ids + ([self.tokenizer.pad_token] * (pad_len - 1))


        # to tensor
        qae_len = torch.zeros(3, dtype=torch.long)
        qae_len[0] = q_len
        qae_len[1] = a_len
        qae_len[2] = e1_len
        label_q_a = [-100] * (q_len + 2) + self.tokenizer.convert_tokens_to_ids(q_a_ids[q_len + 2:q_a_len]) + [-100] * (
                    len(q_a_ids) - q_a_len)

        label_q_e1 = [-100] * (q_len + 2) + self.tokenizer.convert_tokens_to_ids(q_e1_ids[q_len + 2:q_e1_len]) + [-100] * (
                    len(q_e1_ids) - q_e1_len)


        label_q_e1_a = [-100] * (q_len + e1_len + 2) + self.tokenizer.convert_tokens_to_ids(
            q_e1_a_ids[q_len + e1_len + 2:q_e1_a_len]) + [-100] * (
                              len(q_e1_a_ids) - q_e1_a_len)


        label_q_a = torch.tensor(label_q_a, dtype=torch.long)

        label_q_e1 = torch.tensor(label_q_e1, dtype=torch.long)
        label_q_e1_a = torch.tensor(label_q_e1_a, dtype=torch.long)


        q_a_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(q_a_ids), dtype=torch.long)
        q_e1_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(q_e1_ids), dtype=torch.long)
        q_e1_a_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(q_e1_a_ids), dtype=torch.long)


        a = len(q_e1_ids)
        if len(q_e1_ids) > self.max_seq_len:
            q_e1_ids = q_e1_ids[:self.max_seq_len]
            label_q_e1 = label_q_e1[:self.max_seq_len]
        if len(q_e1_a_ids) > self.max_seq_len:
            q_e1_a_ids = q_e1_a_ids[:self.max_seq_len]
            label_q_e1_a = label_q_e1_a[:self.max_seq_len]

        if len(q_a_ids) > self.max_seq_len:
            q_a_ids =q_a_ids[:self.max_seq_len]
            label_q_a=label_q_a[:self.max_seq_len]
            print(self.tokenizer.decode(q_a_ids))
            print("wo cao!")
        q_a = {'input_ids': q_a_ids,
               'len': q_a_len,
               'label': label_q_a}

        q_e1 = {'input_ids': q_e1_ids,
               'len': q_e1_len,
               'label': label_q_e1}

        q_e1_a = {
            'input_ids': q_e1_a_ids,
            'len': q_e1_a_len,
            'label': label_q_e1_a
        }

        ans = prompt_answer + '<|endoftext|>'


        folder = '../aokvqa/coco2017/train2017/'
        img_path = folder +f"{img_id:012}.jpg"
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)


        return (qae_len , img,q_a, q_e1,q_e1_a, prompt_q , ans)

    def __len__(self):
        return self.ids_list
class AOKEvalDataset(Dataset):

    def __init__(self, path, transform, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_seq_len = max_seq_len  # question + <bos> The answer is <answer> becase <explanation> <eos>
        self.data = json.load(open(path, 'r'))
        self.ids_list = len(self.data)

    def __getitem__(self, i):
        quention_id = self.data[i]['question_id']
        sample = self.data[i]
        img_id = sample['image_id']
        text_a = data_utils.proc_ques(sample['question'])  # question

        prompt_a = ' question: ' + text_a + ' context:'

        tokens = self.tokenizer.tokenize(prompt_a)

        q_c = self.tokenizer.convert_tokens_to_ids(tokens)
        q_c = torch.tensor(q_c, dtype=torch.long)

        img_path = os.path.join("../aokvqa/data/coco2017", f"val2017", f"{img_id:012}.jpg")
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        qid = quention_id
        return (img, qid, q_c )

    def __len__(self):
        return self.ids_list