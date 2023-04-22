import json
with open('AOKVQA/aokvqa/aokvqa_v1p0_train.json','r') as f:
    aokvqa = json.load(f)
with open('vqax/vqaX_test.json', 'r') as f:
    vqax = json.load(f)
vqax2aokvqa = []
for key,value in vqax.items():
    dict = {}
    dict['split'] = 'train'
    dict['question_id'] = key
    dict['question'] = value['question']
    dict['answers'] = [a['answer'] for a in value['answers']]
    dict['image_id'] = value['image_name']
    dict['rationales'] = value['explanation']
    vqax2aokvqa.append(dict)
with open('vqax/vqaX_test_va.json','w') as f:
    json.dump(vqax2aokvqa,f)
print ('afds')