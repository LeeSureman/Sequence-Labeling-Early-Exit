#BERT跑英文ner

import sys
sys.path.append('..')
from V0.load_data_ import load_conll,load_ontonotes_cn,load_conll_two_col
import torch
from fastNLP import Trainer
from fastNLP import LossInForward
from fastNLP import BucketSampler
from fastNLP import FitlogCallback,EarlyStopCallback
import fitlog
# from fastNLP import AccuracyMetric
from tmp_fastnlp_module import MySpanFPreRecMetric as SpanFPreRecMetric,MyAccuracyMetric as AccuracyMetric,MyWarmupCallback as WarmupCallback

# fitlog.debug()
fitlog.set_log_dir('logs')
# a = Vocabulary()
# a.unknown_idx
# from fastNLP.embeddings.ro
# from fastNLP
import argparse
from paths import *
import torch.optim as optim


parser = argparse.ArgumentParser()

parser.add_argument('--device',default=0,)
parser.add_argument('--demo',type=int,default=0)
parser.add_argument('--demo_train',type=int,default=0)
parser.add_argument('--debug',type=int,default=0)
parser.add_argument('--dataset',choices=['conll','ontonotes','ontonotes_cn','ctb5_pos','ctb7_pos','ctb9_pos','weibo','e_com','msra_ner',
                                         'ud1_pos','ud2_pos','ctb5_seg','ud_seg','twitter_ner','ark_twitter_pos','ritter_pos','clue_ner'],default='ud_seg')
parser.add_argument('--encoding_type',choices=['bioes','bio'],default='bio')
parser.add_argument('--seed',type=int,default=100)
parser.add_argument('--use_pytorch_dropout',default=False)
parser.add_argument('--msg')
parser.add_argument('--use_fastnlp_bert',type=int,default=0)
parser.add_argument('--if_save',type=int,default=0)

parser.add_argument('--train_mode',choices=['one_cls','joint','joint_two_stage'],default='joint_two_stage')
parser.add_argument('--test_mode',choices=['one_cls','joint','s_level_ee','t_level_ee','t_level_ee_copy'],default='joint')


# parser.add_argument('--ptm_dropout',type=float,default=0.5)
parser.add_argument('--use_crf',default=0,type=int)
parser.add_argument('--ptm_name',default='hfl/chinese-bert-wwm',
                    choices=['albert-base-v2','hfl/chinese-bert-wwm','bert-base-cased']) #适用于transformers，而不是FastNLP

parser.add_argument('--fast_ptm_name',choices=['roberta','bert','albert',''],default='bert')
parser.add_argument('--after_bert',choices=['linear','tener'],default='tener')
# parser.add_argument('--ptm_layers',default='-1')
parser.add_argument('--ptm_pool_method', choices=['first', 'last', 'first_skip_space'], default='first')
parser.add_argument('--cls_hidden',type=int,default=128)
parser.add_argument('--cls_head',type=int,default=2)
parser.add_argument('--cls_ff',type=int,default=2)
parser.add_argument('--cls_dropout',type=float,default=0.3)
parser.add_argument('--cls_after_norm',type=int,default=1)
parser.add_argument('--cls_scale',type=int,default=0)
parser.add_argument('--cls_drop_attn',type=float,default=0.05)
parser.add_argument('--use_bigram',type=int,default=1)
parser.add_argument('--use_char',type=int,default=1)
parser.add_argument('--use_word',type=int,default=0) # can try
parser.add_argument('--embed_dropout',type=float,default=0.3)
parser.add_argument('--keep_norm_same',type=int,default=0)
parser.add_argument('--word_embed_dim',type=int,default=100)
# parser.add_argument('--ptm_word_dropout',default=0.01,type=float)

parser.add_argument('--sampler',default='bucket',choices=['bucket','sequential','random'])
parser.add_argument('--batch_size',type=int,default=10)
parser.add_argument('--epoch',type=int,default=20)
parser.add_argument('--optimizer',default='adam',choices=['adam','sgd'])
parser.add_argument('--lr',type=float,default=1e-4) # bert crf 2e-4, bert w/o crf
parser.add_argument('--ptm_lr_rate',type=float,default=0.1)
parser.add_argument('--crf_lr_rate',type=float,default=1)
parser.add_argument('--momentum',type=float,default=0.9)
parser.add_argument('--weight_decay',type=float,default=1e-2)
parser.add_argument('--warmup_step',type=float,default=3000)
parser.add_argument('--warmup_schedule',choices=['linear','constant','inverse_square'],default='linear')
parser.add_argument('--early_stop_patience',type=int,default=-1)
parser.add_argument('--fix_ptm_epoch',type=int,default=2)
parser.add_argument('--fix_ptm_step',type=int,default=-1)
parser.add_argument('--gradient_clip_norm_bert',type=float,default=1)
parser.add_argument('--gradient_clip_norm_other',type=float,default=5) # english 15 , chinese 5
parser.add_argument('--joint_weighted',type=int,default=1)
parser.add_argument('--cls_common_lr_scale',type=int,default=1)


args = parser.parse_args()

if args.fix_ptm_epoch>0 and args.fix_ptm_step>0:
    print('only fix one, epoch or step! exit!')
    exit(1)

en_dataset = ['conll','ontonotes','twitter_ner','ark_twitter_pos','ritter_pos']
cn_dataset = ['ontonotes_cn','weibo','e_com','ctb5_pos','ctb7_pos','ctb9_pos','msra_ner','ud1_pos','ud2_pos','ud_seg','ctb5_seg','clue_ner']
en_ptm = ['albert-base-v2','bert-base-cased']
cn_ptm = ['hfl/chinese-bert-wwm']
language_ = 'none'
if args.dataset in en_dataset:
    language_ = 'en'
    if args.fast_ptm_name == '':
        assert args.ptm_name in en_ptm
elif args.dataset in cn_dataset:
    language_ = 'cn'
    if args.fast_ptm_name == '':
        assert args.ptm_name in cn_ptm
else:
    print('language uncertain! exit!')
    exit()

if args.fast_ptm_name != '':
    if args.fast_ptm_name == 'bert':
        if language_ == 'en':
            args.ptm_name = 'bert-base-cased'
        else:
            args.ptm_name = 'hfl/chinese-bert-wwm'
    elif args.fast_ptm_name == 'roberta':
        raise NotImplementedError
        if language_ == 'en':
            args.ptm_name = 'roberta_en'
        else:
            print('not support roberta cn')
            raise NotImplementedError
    elif args.fast_ptm_name == 'albert':
        # raise NotImplementedError
        if language_ == 'en':

            args.ptm_name = 'albert-base-v2'
        else:
            raise NotImplementedError


args.language_ = language_
fitlog.set_rng_seed(args.seed)
fitlog.add_hyper(args)
train_mode = args.train_mode

if args.device not in ['cpu','all']:
    assert isinstance(args.device,int) or args.device.isdigit()
    device = torch.device('cuda:{}'.format(args.device))
elif args.device == 'cpu':
    device = torch.device('cpu')
elif args.device == 'all':
    device = [i for i in range(torch.cuda.device_count())]
else:
    raise NotImplementedError

if args.demo or args.demo_train:
    fitlog.debug()

if args.demo and args.demo_train:
    print(args.demo)
    print(args.demo_train)
    print('demo 和 demo_train 不能同时开，所以退出 exit')
    exit()

if args.device!='cpu':
    assert isinstance(args.device,int) or args.device.isdigit()
    device = torch.device('cuda:{}'.format(args.device))
else:
    device = torch.device('cpu')



refresh_data = False

cache_name = 'cache/{}_{}_{}'.format(args.dataset,args.encoding_type,args.ptm_name)
if args.dataset == 'ontonotes':
    raise NotImplementedError
    bundle = load_ontonotes(ontonotes_path,args.encoding_type,_cache_fp=cache_name)
elif args.dataset == 'conll':
    bundle = load_conll(conll_path,args.encoding_type,pretrained_model_name_or_path=args.ptm_name, _cache_fp=cache_name,_refresh=refresh_data)
elif args.dataset == 'ontonotes_cn':
    bundle = load_ontonotes_cn(ontonote4ner_cn_path,args.encoding_type,pretrained_model_name_or_path=args.ptm_name,
                               _cache_fp=cache_name,_refresh=refresh_data,dataset_name=args.dataset)
elif args.dataset == 'ctb5_pos':
    bundle = load_ontonotes_cn(ctb5_char_path,args.encoding_type,pretrained_model_name_or_path=args.ptm_name,
                               _cache_fp=cache_name,_refresh=refresh_data,dataset_name=args.dataset)
elif args.dataset == 'ctb7_pos':
    bundle = load_ontonotes_cn(ctb7_char_path,args.encoding_type,pretrained_model_name_or_path=args.ptm_name,
                               _cache_fp=cache_name,_refresh=refresh_data,dataset_name=args.dataset)
elif args.dataset == 'ctb9_pos':
    bundle = load_ontonotes_cn(ctb9_char_path,args.encoding_type,pretrained_model_name_or_path=args.ptm_name,
                               _cache_fp=cache_name,_refresh=refresh_data,dataset_name=args.dataset)
elif args.dataset == 'ctb5_seg':
    bundle = load_ontonotes_cn(ctb5_char_path,args.encoding_type,pretrained_model_name_or_path=args.ptm_name,
                               _cache_fp=cache_name,_refresh=refresh_data,dataset_name=args.dataset)
elif args.dataset == 'weibo':
    bundle = load_ontonotes_cn(weibo_ner_path,args.encoding_type,pretrained_model_name_or_path=args.ptm_name,
                               _cache_fp=cache_name,_refresh=refresh_data,dataset_name=args.dataset)
elif args.dataset == 'e_com':
    bundle = load_ontonotes_cn(ecom_ner_path,args.encoding_type,pretrained_model_name_or_path=args.ptm_name,
                               _cache_fp=cache_name,_refresh=refresh_data,dataset_name=args.dataset)
elif args.dataset == 'msra_ner':
    bundle = load_ontonotes_cn(msra_ner_cn_path,args.encoding_type,pretrained_model_name_or_path=args.ptm_name,
                               _cache_fp=cache_name,_refresh=refresh_data,dataset_name=args.dataset)
elif args.dataset == 'ud1_pos':
    bundle = load_ontonotes_cn(ud1_path,args.encoding_type,pretrained_model_name_or_path=args.ptm_name,
                               _cache_fp=cache_name,_refresh=refresh_data,dataset_name=args.dataset)
elif args.dataset == 'ud2_pos':
    bundle = load_ontonotes_cn(ud2_path,args.encoding_type,pretrained_model_name_or_path=args.ptm_name,
                               _cache_fp=cache_name,_refresh=refresh_data,dataset_name=args.dataset)
elif args.dataset == 'ud_seg':
    bundle = load_ontonotes_cn(ud1_path,args.encoding_type,pretrained_model_name_or_path=args.ptm_name,
                               _cache_fp=cache_name,_refresh=refresh_data,dataset_name=args.dataset)
elif args.dataset == 'clue_ner':
    bundle = load_ontonotes_cn(clue_ner_path,args.encoding_type,pretrained_model_name_or_path=args.ptm_name,
                               _cache_fp=cache_name,_refresh=refresh_data,dataset_name=args.dataset)
elif args.dataset == 'twitter_ner':
    print('fjl_twitter_ner_path:{}'.format(fjl_twitter_ner_path))
    bundle = load_conll_two_col(fjl_twitter_ner_path,args.encoding_type,pretrained_model_name_or_path=args.ptm_name, _cache_fp=cache_name,_refresh=refresh_data)
elif args.dataset == 'ritter_pos':
    bundle = load_conll_two_col(ritter_path,args.encoding_type,pretrained_model_name_or_path=args.ptm_name, _cache_fp=cache_name,_refresh=refresh_data)
elif args.dataset == 'ark_twitter_pos':
    bundle = load_conll_two_col(ark_twitter_path,args.encoding_type,pretrained_model_name_or_path=args.ptm_name, _cache_fp=cache_name,_refresh=refresh_data)
else:
    print('不支持该数据集')
    exit()

# for k,v in bundle.datasets.items():
#     for ins in v:
#         if ins['word_piece_seq_len']>500:
#             print('big! {}'.format(ins['word_piece_seq_len']))
#
# exit()

if args.demo_train:
    # print(bundle.datasets['train'][0]['words'])
    bundle.datasets['train'] = bundle.datasets['train'][:500]
    bundle.datasets['dev'] = bundle.datasets['train']
    bundle.datasets['test'] = bundle.datasets['train']

if args.demo:
    from utils import shuffle_dataset

    bundle.datasets['train'] = shuffle_dataset(bundle.datasets['train'])
    bundle.datasets['train'] = bundle.datasets['train'][:100]

# a = DataSet()
# a = Vocabulary()
# a.padding_idx
# a.set_pad_val(field_name=)

for k,v in bundle.datasets.items():
    v.set_pad_val('words',bundle.vocabs['words'].padding_idx)


# dropout_dict = {}
# dropout_dict['bert'] = args.bert_dropout

from V0.modeling_bert_ee import BERT_SeqLabel_EE
model = BERT_SeqLabel_EE(bundle,args)
print('*'*20,'param not ptm','*'*20)
for k,v in model.named_parameters():
    if 'ptm_encoder.' not in k:
        print('{}:{}'.format(k,v.size()))
print('*' * 20, 'param not ptm', '*' * 20)

# model_param = list(model.parameters())
# ptm_param = list(model.ptm_encoder.parameters())
# ptm_param_id = list(map(id,ptm_param))
# for name, param in model.named_parameters():
#     if ''

# non_ptm_param = list(filter(lambda x:id(x) not in ptm_param_id,model_param))

params = {}
params['crf'] = []
params['ptm'] = []
params['other'] = []
params['basic_embedding'] = []

params_name = {}
params_name['crf'] = []
params_name['ptm'] = []
params_name['other'] = []
params_name['basic_embedding'] = []
# for k,v

#1
# params = {}
# params['crf'] = []
# params['ptm'] = []
# params['other'] = []

# for k,v

# for k,v in model.named_parameters():
#     # print(k,v)
#     if k[:len('ptm_encoder.')] == 'ptm_encoder.':
#         params['ptm'].append(v)
#     elif k[:len('crf.')] == 'crf.':
#         params['crf'].append(v)
#     else:
#         params['other'].append(v)

# exit()

# param_ = [{'params':params['ptm'],'lr':args.lr*args.ptm_lr_rate},
#           {'params':params['crf'],'lr':args.lr*args.crf_lr_rate},
#           {'params':params['other'],'lr':args.lr}]
#2

for k,v in model.named_parameters():
    # print(k,v.size())
    if k[:len('ptm_encoder.')] == 'ptm_encoder.':
        params['ptm'].append(v)
        params_name['ptm'].append(k)
    elif 'cnn_char.' in k or 'bigram_embedding.' in k or 'word_embedding.' in k:
        params['basic_embedding'].append(v)
        params_name['basic_embedding'].append(k)

    elif k[:len('crf.')] == 'crf.':
        params['crf'].append(v)
        params_name['crf'].append(k)
    else:
        params['other'].append(v)
        params_name['other'].append(k)

# exit()
if args.train_mode == 'one_cls':
    param_ = [{'params':params['ptm'],'lr':args.lr*args.ptm_lr_rate},
              {'params':params['crf'],'lr':args.lr*args.crf_lr_rate},
              {'params':params['other'],'lr':args.lr},
              {'params':params['basic_embedding'],'lr':args.lr}]
elif 'joint' in args.train_mode:
    if False:
        raise NotImplementedError
        pass
    else:
        if args.cls_common_lr_scale:
            if args.joint_weighted:
                raise NotImplementedError
                pass
            # num_layers = 1
            if 'large' not in args.ptm_name:
                num_layers = 12
            else:
                raise NotImplementedError
        else:
            num_layers = 1
        param_ = [{'params': params['ptm'], 'lr': args.lr * args.ptm_lr_rate},
                  {'params': params['crf'], 'lr': args.lr * args.crf_lr_rate},
                  {'params': params['other'], 'lr': args.lr*num_layers},
                  {'params': params['basic_embedding'], 'lr': args.lr}]

#3

if args.optimizer == 'adam':
    optimizer = optim.AdamW(param_,lr=args.lr,weight_decay=args.weight_decay)
elif args.optimizer == 'sgd':
    optimizer = optim.SGD(param_,lr=args.lr,momentum=args.momentum,
                          weight_decay=args.weight_decay)

callbacks = []
# callbacks.append(FitlogCallback)
from fastNLP import GradientClipCallback
gradient_callback_bert = GradientClipCallback(params['ptm'],clip_value=args.gradient_clip_norm_bert,clip_type='norm')
gradient_callback_other = GradientClipCallback(params['other'],clip_value=args.gradient_clip_norm_other,clip_type='norm')

callbacks.append(gradient_callback_bert)
callbacks.append(gradient_callback_other)



if args.warmup_step:
    callbacks.append(WarmupCallback(warmup=args.warmup_step,schedule=args.warmup_schedule))
if args.fix_ptm_epoch>0:
    from utils import Unfreeze_Callback
    # model.ptm_encoder.requires_grad = False
    for k,v in model.ptm_encoder.named_parameters():
        v.requires_grad = False
    print('model param freeze!')
    # exit()
    callbacks.append(Unfreeze_Callback(model.ptm_encoder,fix_epoch_num=args.fix_ptm_epoch))
elif args.fix_ptm_step>0:
    from utils import Unfreeze_Callback
    # model.ptm_encoder.requires_grad = False
    for k,v in model.ptm_encoder.named_parameters():
        v.requires_grad = False
    print('model param freeze!')
    # exit()
    callbacks.append(Unfreeze_Callback(model.ptm_encoder,fix_step_num=args.fix_ptm_step))
else:
    if hasattr(model.ptm_encoder,'requires_grad'):
        assert model.ptm_encoder.requires_grad
if args.early_stop_patience>0:
    callbacks.append(EarlyStopCallback(args.early_stop_patience))

metrics = []
# acc_metric = AccuracyMetric(pred='pred',target='target')
# acc_metric.set_metric_name('acc')
english_pos_dataset = ['ritter_pos','ark_twitter_pos']
if args.train_mode == 'one_cls':
    if args.dataset not in english_pos_dataset:
        f_metric = SpanFPreRecMetric(-1,bundle.vocabs['target'],'pred','target',encoding_type=args.encoding_type)
        f_metric.set_metric_name('l_12')
        metrics.append(f_metric)
    acc_metric = AccuracyMetric(-1,pred='pred',target='target')
    acc_metric.set_metric_name('acc_12')
    metrics.append(acc_metric)


    if args.dataset not in english_pos_dataset:
        nb_f_metric = SequentialSpanFMetric(bundle.vocabs['target'], 'pred', 'target', encoding_type=args.encoding_type)
        nb_f_metric.set_metric_name('l12_nb')
        metrics.append(nb_f_metric)
elif 'joint' in args.train_mode:
    i = 11
    f_metric = SpanFPreRecMetric(i, bundle.vocabs['target'], 'pred', 'target', encoding_type=args.encoding_type,is_main_two_stage=1)
    f_metric.set_metric_name('l_{}'.format(i + 1))
    metrics.append(f_metric)

    acc_metric = AccuracyMetric(i, pred='pred', target='target',is_main_two_stage=1)
    acc_metric.set_metric_name('acc_{}'.format(i + 1))
    metrics.append(acc_metric)

    for i in range(11):
        f_metric = SpanFPreRecMetric(i, bundle.vocabs['target'], 'pred', 'target', encoding_type=args.encoding_type,is_main_two_stage=1)
        f_metric.set_metric_name('l_{}'.format(i+1))
        metrics.append(f_metric)

        acc_metric = AccuracyMetric(i,pred='pred', target='target',is_main_two_stage=1)
        acc_metric.set_metric_name('acc_{}'.format(i+1))
        metrics.append(acc_metric)

# metrics.append(acc_metric)

for k,v in bundle.datasets.items():
    v.set_input('words','seq_len','target','word_pieces','bert_attention_mask')
    if language_ == 'cn':
        if args.use_bigram:
            v.set_input('bigrams')
    if args.ptm_pool_method == 'first':
        v.set_input('first_word_pieces_pos')
    elif args.ptm_pool_method == 'first_skip_space':
        v.set_input('first_word_pieces_pos_skip_space')
    elif args.ptm_pool_method == 'last':
        v.set_input('last_word_pieces_pos')

    v.set_pad_val('bert_attention_mask',pad_val=0)
    if args.use_crf:
        v.set_pad_val('target',pad_val=0)
    else:
        v.set_pad_val('target',pad_val=-100)
    v.set_pad_val('first_word_pieces_pos',pad_val=0)
    v.set_pad_val('first_word_pieces_pos_skip_space',pad_val=0)
    v.set_pad_val('last_word_pieces_pos',pad_val=0)

    v.set_target('target','seq_len')


# print(list(bundle.datasets['test'].get_all_fields()))
# exit()
# DataSet

# for k,v in bundle.datasets.items():
#     input_f_name = list(v.get_all_fields().keys())
#     for name in input_f_name:
#         if name in v.field_arrays:
#             print('{},{}:{}'.format(k,name,v.field_arrays[name].is_input))
# exit()


# with torch.no_grad():
#     if args.after_bert in ['tener']:
#         print('{}init pram{}'.format('*'*15,'*'*15))
#         for n,p in model.transformer_cls.named_parameters():
#             if 'embedding' not in n and 'pos' not in n and 'pe' not in n \
#                     and 'bias' not in n and 'crf' not in n and p.dim()>1:
#                 try:
#                     nn.init.xavier_uniform_(p)
#                     print('xavier uniform init:{}'.format(n))
#                     # if args.init == 'uniform':
#                     #     nn.init.xavier_uniform_(p)
#                     #     print_info('xavier uniform init:{}'.format(n))
#                     # elif args.init == 'norm':
#                     #     print_info('xavier norm init:{}'.format(n))
#                     #     nn.init.xavier_normal_(p)
#                 except Exception as e:
#                     print(e)
#                     print(n)
#                     exit(1208)
#         print('{}init pram{}'.format('*' * 15, '*' * 15))
from fastNLP import Callback
class Save_Model_Callback(Callback):
    def __init__(self,model,output_path):
        super().__init__()
        self.model_ = model
        self.output_path = output_path

    def on_valid_end(self, eval_result, metric_key, optimizer, is_better_eval):
        if True:
            f_model = open('{}/model/epoch_{}.pkl'.format(self.output_path,self.epoch),'wb')
            torch.save(self.model_,f_model)
            f_model.close()




from fastNLP import SequentialSampler

# tester = Tester(bundle.datasets['train'],model,metrics,batch_size=4,device=device)
# tester.test()
# exit()
# print(model.ptm_encoder.requires_grad)
# exit()
# nn.Linear.weight
# print(model.ptm_encoder.encoder.layer[11].attention.self.query.weight[:10,10])
# exit()
if args.sampler == 'bucket':
    sampler = BucketSampler()
elif args.sampler  == 'sequential':
    sampler = SequentialSampler()
# if train_mode ==
# from fastNLP import Callback
# class Modify_Train_Mode_Callback(Callback):
#     def __init__(self,args,modify_epoch,modify_train_mode,fix_param,un_fix_param):
#         super().__init__()
#         self.args = args
#         self.modify_epoch = modify_epoch
#         self.modify_train_mode = modify_train_mode
#         assert modify_train_mode == 'joint'
#         self.fix_param = fix_param
#
#     def on_epoch_begin(self):
#         if self.epoch == self.modify_epoch:
#             self.args.train_mode = self.modify_train_mode
#
#             for param in self.fix_param:
#                 param.requires_grad = False
#
#             for param in self.unfix_param:
#                 param.requires_grad = True


# modify_train_mode_callback = Modify_Train_Mode_Callback(args,20,'joint',fix_param,unfix_param)
# callbacks.append(modify_train_mode_callback)
args.train_mode = 'one_cls'
import copy
callbacks_copy = copy.deepcopy(callbacks)

trainer = Trainer(bundle.datasets['train'],model,optimizer,LossInForward(),args.batch_size,sampler,
                  n_epochs=5,
                  dev_data=bundle.datasets.get('dev') if 'dev' in bundle.datasets else bundle.datasets['test'],
                  metrics=metrics,metric_key='f' if args.dataset not in english_pos_dataset else None,
                  callbacks=callbacks,test_use_tqdm=False,device=device)
trainer.train()

fix_param = []

for param in model.parameters():
    fix_param.append(param)

unfix_param = []
for cls in model.classfiers[:-1]:
    for param in cls.parameters():
        unfix_param.append(param)



callbacks = callbacks_copy

if bundle.datasets.get('dev') is not None:
    if bundle.datasets.get('test') is not None:
        fitlog_callback = FitlogCallback(data={'train':bundle.datasets['train'][:2000],'test':bundle.datasets['test']},log_loss_every=10)
    else:
        fitlog_callback = FitlogCallback(data={'train':bundle.datasets['train'][:2000]},log_loss_every=10)
else:
    fitlog_callback = FitlogCallback(data={'train':bundle.datasets['train'][:2000]}, log_loss_every=10)

callbacks.append(fitlog_callback)

if args.if_save:
    from utils import get_peking_time
    exp_tag = get_peking_time()
    fitlog.add_other(str(exp_tag)[5:],'time_tag')
    exp_path = 'exps/{}'.format(exp_tag)
    import os
    os.makedirs(exp_path,exist_ok=False)
    os.makedirs('{}/model'.format(exp_path), exist_ok=False)
    f_bundle = open('{}/bundle.pkl'.format(exp_path),'wb')
    f_args = open('{}/args.pkl'.format(exp_path),'wb')
    torch.save(bundle,f_bundle)
    torch.save(args,f_args)

    f_bundle.close()
    f_args.close()

    callbacks.append(Save_Model_Callback(model,exp_path))


args.train_mode = 'joint_two_stage'
trainer = Trainer(bundle.datasets['train'],model,optimizer,LossInForward(),args.batch_size,sampler,
                  n_epochs=args.epoch,
                  dev_data=bundle.datasets.get('dev') if 'dev' in bundle.datasets else bundle.datasets['test'],
                  metrics=metrics,metric_key='f' if args.dataset not in english_pos_dataset else None,
                  callbacks=callbacks,test_use_tqdm=False,device=device)
trainer.train()




fitlog.finish()

