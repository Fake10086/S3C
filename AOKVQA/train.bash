CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
nohup /home/suowei/anaconda3/envs/py3.7-torch1.8/bin/python -u -m torch.distributed.launch \
--nproc_per_node=8 --master_port=11111 --use_env train/clipcap_train.py \
--log-dir \
logs \
--aokvqa-dir \
aokvqa \
--train-features \
../features/clip-ViT-B-32_train.pt \
--val-features \
../features/clip-ViT-B-32_val.pt \
--pretrained-model \
../pretrain_model/coco_weights.pt \
--generation-target \
rationale \
--mapping \
mlp \
--finetune-gpt \
