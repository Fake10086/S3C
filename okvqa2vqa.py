import json
with open('okvqa/mscoco_train2014_annotations.json','r') as f:
    anno = json.load(f)['annotations']
with open('okvqa/OpenEnded_mscoco_train2014_questions.json','r') as f:
    ques = json.load(f)['questions']
with open('vqa/train.json','r') as f:
    vqa = json.load(f)
okvqa = []
def label_extract(answer):
    dict = {}
    for a in answer:
        if a['answer'] not in dict:
            dict[a['answer']]=1
        else:
            dict[a['answer']] = dict[a['answer']]+1
    a = sorted(dict.items(), key=lambda kv: (kv[1], kv[0]))[-1]
    aa = {a[0]:1}
    return aa
for ann,que in zip(anno,ques):
    dict = {}
    dict['answer_type'] = ann['answer_type']
    dict['img_id'] = 'COCO_train2014_'+(12-len(str(ann['image_id'])))*str(0)+str(ann['image_id'])
    label = [a['answer'] for a in ann['answers']]
    dict['label'] = label
    dict['sent'] = que['question']
    dict['question_id'] = que['question_id']
    okvqa.append(dict)
with open('okvqa/train.json','w') as f:
    json.dump(okvqa,f)
print ('adf')
