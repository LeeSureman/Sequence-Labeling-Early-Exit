# Sequence-Labeling-Early-Exit
Code for ACL 2021 paper: Accelerating BERT Inference for Sequence Labeling via Early-Exit


# Requirement:
Please refer to requirements.txt

# How to run?

For ontonotes (CN):

you should claim your dataset path in paths.py, and then

```angular2html
For the first stage training:

python -u main.py --device 0  --seed 100 --fast_ptm_name bert --lr 5e-5  --use_crf 0 --dataset ontonotes_cn --fix_ptm_epoch 2 --warmup_step 3000 --use_fastnlp_bert 0 --sampler bucket  --after_bert linear --use_char 0 --use_bigram 0 --gradient_clip_norm_other 5 --gradient_clip_norm_bert 1 --train_mode joint --test_mode joint --if_save 1 --warmup_schedule inverse_square --epoch 20 --joint_weighted 1 --ptm_lr_rate 0.1 --cls_common_lr_scale 0

Then find the exp_path in the corresponding fitlog entry, and self-sampling further train the model.

For the self-sampling training:

python -u further_train.py --seed 100 --msg fuxian --if_save 1 --warmup_schedule inverse_square --epoch 30 --keep_norm_same 1 --sandwich_small 2 --sandwich_full 4 --max_t_level_t -0.5 --train_mode joint_sample_copy --further 0 --flooding 1 --flooding_bias 0 --lr 1e-4 --ptm_lr_rate 0.1 --fix_ptm_epoch 2 --min_win_size 5 --copy_wordpiece all --ckpt_epoch 7 --exp_path 05_11_22_20_52.210103 --device 2 --max_threshold 0.25 --max_threshold_2 0.5

Then find the exp_path and best epoch in the corresponding fitlog entry, and use it for early-exit inference as:

speed 2X:
python test.py --device 2 --further 1 --record_flops 1 --win_size 15 --threshold 0.1 --ckpt_epoch [ckpt_path] --exp_path [exp_path]
speed 3X:
python test.py --device 2 --further 1 --record_flops 1 --win_size 5 --threshold 0.15 --ckpt_epoch [ckpt_path] --exp_path [exp_path]
speed 4X:
python test.py --device 2 --further 1 --record_flops 1 --win_size 5 --threshold 0.25 --ckpt_epoch [ckpt_path] --exp_path [exp_path]


```
Other datasets' scripts coming soon

If you have any question, do not hesitate to ask it in issue. (English or Chinese both ok) 