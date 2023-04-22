import json
import  torch

with open('vqax/vqaX_train_va.json', 'r') as f:
    aokvqa_set = json.load(f)
qqq = []
for i in aokvqa_set:
    if ' '.join(i['question'].split()[:5]) not in qqq:
        qqq.append(' '.join(i['question'].split()[:5]))


lv_chu = ['what color is the',
          'what is this',
          'what is in the',
          'what is the',
          'how many',
          'what color is',
          'what is on the',
          'what is',
          'how many people are in',
          'is this a',
          'how many people are',
          'what color are the',
          'what is the color of the',
          'what color',
          ]
with open('vqa/train.json','r') as f:
    a = json.load(f)
question = []
idx = 1
for i in a:
    if ' '.join(i['sent'].split()[:5]) in qqq:
        idx = idx+1
    #     question.append(i['question_type'])
    # if i['question_type'] in lv_chu:
    #     continue
    else:
        continue

print ('adsf')