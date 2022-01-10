import sys
sys.path.append('..')
from fastNLP import Tester
import argparse
import torch
from tmp_fastnlp_module import MySpanFPreRecMetric as SpanFPreRecMetric,MyAccuracyMetric as AccuracyMetric
import fitlog
fitlog.set_log_dir('logs_further')
import torch.optim as optim
from fastNLP import Trainer, LossInForward
from tmp_fastnlp_module import MySpanFPreRecMetric as SpanFPreRecMetric,MyAccuracyMetric as AccuracyMetric,MyWarmupCallback as WarmupCallback
from fastNLP import FitlogCallback
import torch.nn as nn

parser = argparse.ArgumentParser()
# parser.add_argument('--exp_path',default='01_11_16_52_01.631786')
parser.add_argument('--exp_path',default='01_20_14_48_37.579284')
parser.add_argument('--ckpt_epoch',type=int,default=11)
parser.add_argument('--device',default='0')
parser.add_argument('--flooding',type=int,default=0)
parser.add_argument('--test_joint',type=int,default=0)
parser.add_argument('--train_part',type=int,default=-1)
# parser.add_argument('--criterion',choices=['entropy','max_p'],default='entropy')
# parser.add_argument('--sentence_ee_pool',choices=['avg','max'],default='max')
# parser.add_argument('--threshold',type=float,default=0.1)
# parser.add_argument('--test_mode',choices=['one_cls','joint','s_level_ee','t_level_ee','t_level_ee_copy','t_level_win_ee'],default='t_level_win_ee')
# parser.add_argument('--win_size',type=int,default=11)
# parser.add_argument('--test_part',type=float,default=-1)







####
# parser.add_argument('--device',default=0,)
parser.add_argument('--demo',type=int,default=0)
parser.add_argument('--demo_train',type=int,default=0)
parser.add_argument('--debug',type=int,default=0)
parser.add_argument('--seed',type=int,default=100)
parser.add_argument('--msg',default='two_stage')
parser.add_argument('--if_save',type=int,default=1)
parser.add_argument('--true_copy',type=int,default=1)
parser.add_argument('--true_label',type=int,default=1)
parser.add_argument('--unk_label_embed_zero',type=int,default=1)
parser.add_argument('--label_embed_use_last_cls',type=int,default=1)

parser.add_argument('--train_mode',choices=[
    # 'one_cls','joint','joint_sample_copy','joint_sample_copy_label',
    'joint_two_stage'],
                    default='joint_two_stage')
parser.add_argument('--copy_supervised',type=int,default=1)
parser.add_argument('--test_mode',choices=['one_cls','joint','s_level_ee','t_level_ee','t_level_ee_copy'],default='joint')

parser.add_argument('--min_win_size',type=int,default=-1)
parser.add_argument('--max_win_size',type=int,default=2000)

parser.add_argument('--min_threshold',type=float,default=-1)
parser.add_argument('--max_threshold',type=float,default=1000)

parser.add_argument('--min_t_level_t',type=float,default=-1)
parser.add_argument('--max_t_level_t',type=float,default=1000)

parser.add_argument('--sandwich_small',type=int,default=2)
parser.add_argument('--sandwich_full',type=int,default=3)

# parser.add_argument('--ptm_dropout',type=float,default=0.5)
parser.add_argument('--use_crf',default=0,type=int)
# parser.add_argument('--ptm_name',default='hfl/chinese-bert-wwm',
#                     choices=['albert-base-v2','hfl/chinese-bert-wwm','bert-base-cased']) #适用于transformers，而不是FastNLP

# parser.add_argument('--fast_ptm_name',choices=['roberta','bert','albert',''],default='bert')
# parser.add_argument('--after_bert',choices=['linear','tener'],default='tener')
# parser.add_argument('--ptm_layers',default='-1')
# parser.add_argument('--ptm_pool_method', choices=['first', 'last', 'first_skip_space'], default='first')
# parser.add_argument('--cls_hidden',type=int,default=128)
# parser.add_argument('--cls_head',type=int,default=2)
# parser.add_argument('--cls_ff',type=int,default=2)
# parser.add_argument('--cls_dropout',type=float,default=0.3)
# parser.add_argument('--cls_after_norm',type=int,default=1)
# parser.add_argument('--cls_scale',type=int,default=0)
# parser.add_argument('--cls_drop_attn',type=float,default=0.05)
# parser.add_argument('--use_bigram',type=int,default=1)
# parser.add_argument('--use_char',type=int,default=1)
# parser.add_argument('--embed_dropout',type=float,default=0.3)
parser.add_argument('--keep_norm_same',type=int,default=0)
# parser.add_argument('--ptm_word_dropout',default=0.01,type=float)

parser.add_argument('--sampler',default='bucket',choices=['bucket','sequential'])
parser.add_argument('--batch_size',type=int,default=10)
parser.add_argument('--epoch',type=int,default=20)
# parser.add_argument('--epoch',type=int,default=20)

parser.add_argument('--optimizer',default='adam',choices=['adam','sgd'])
parser.add_argument('--lr',type=float,default=2e-4) # bert crf 2e-4, bert w/o crf
parser.add_argument('--ptm_lr_rate',type=float,default=0.1)
parser.add_argument('--crf_lr_rate',type=float,default=10)
parser.add_argument('--momentum',type=float,default=0.9)
parser.add_argument('--weight_decay',type=float,default=1e-2)
# parser.add_argument('--warmup_step',type=float,default=3000)
parser.add_argument('--warmup_step',type=float,default=3000)
# parser.add_argument('--consider_copy_loss',type=int,default=0)
#在训练中，对于已经模拟退出的token，是否还计算loss？对于带label的情况，肯定是不能的，因为会暴露信息，从而影响模型对于label的建模。

parser.add_argument('--warmup_schedule',choices=['linear','constant','inverse_square'],default='inverse_square')
parser.add_argument('--early_stop_patience',type=int,default=-1)
parser.add_argument('--fix_ptm_epoch',type=int,default=-1)
parser.add_argument('--fix_ptm_step',type=int,default=-1)
parser.add_argument('--gradient_clip_norm_bert',type=float,default=1)
parser.add_argument('--gradient_clip_norm_other',type=float,default=15)
parser.add_argument('--joint_weighted',type=int,default=0)
parser.add_argument('--cls_common_lr_scale',type=int,default=0)
####


















args = parser.parse_args()
if args.train_part>0:
    args.if_save = 0
if args.exp_path[:5] == '2021_':
    args.exp_path = args.exp_path[:5]
else:
    pass
    # args.exp_path = '2021_{}'.format(args.exp_path)

if args.device not in ['cpu','all']:
    assert isinstance(args.device,int) or args.device.isdigit()
    device = torch.device('cuda:{}'.format(args.device))
elif args.device == 'cpu':
    device = torch.device('cpu')
elif args.device == 'all':
    device = [i for i in range(torch.cuda.device_count())]
else:
    raise NotImplementedError



model_path = 'exps/2021_{}/model/epoch_{}.pkl'.format(args.exp_path,args.ckpt_epoch)
bundle_path = 'exps/2021_{}/bundle.pkl'.format(args.exp_path)
args_path = 'exps/2021_{}/args.pkl'.format(args.exp_path)

model = torch.load(open(model_path,'rb'),map_location=torch.device('cpu'))
fitlog.add_hyper(model.args.dataset,'dataset')
# model.
# main_args = args
# bundle = torch.load(open(bundle_path,'rb'))
# args = torch.load(open(args_path,'rb'))
# args = torch.load(open(args_path,'rb'))
model_args = model.args

args.encoding_type = model_args.encoding_type
model_args.flooding = args.flooding
model_args.keep_norm_same = args.keep_norm_same
model_args.true_label = args.true_label
model_args.joint_weighted = args.joint_weighted
model_args.copy_supervised = args.copy_supervised

# model.args.test_mode = main_args.test_mode
# model.args.criterion = main_args.criterion
# model.args.sentence_ee_pool = main_args.sentence_ee_pool
# model.args.threshold = main_args.threshold
# model.args.win_size = main_args.win_size

# args.test_mode = main_args.test_mode
model_args.test_mode = args.test_mode

# args.fix_ptm_epoch = main_args.fix_ptm_epoch


bundle = torch.load(open(bundle_path,'rb'))
print('target vocab size:{}'.format(len(bundle.vocabs['target'])))










###
if args.fix_ptm_epoch>0 and args.fix_ptm_step>0:
    print('only fix one, epoch or step! exit!')
    exit(1)




# fitlog.debug()
fitlog.set_rng_seed(args.seed)
fitlog.add_hyper(args)

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
    # bundle.datasets['train'] = bundle.datasets['train'][:100]


for k,v in bundle.datasets.items():
    v.set_pad_val('words',bundle.vocabs['words'].padding_idx)


# from V2.modeling_bert_ee import BERT_SeqLabel_EE
# model = BERT_SeqLabel_EE(bundle,args)
# print('*'*20,'param not ptm','*'*20)
# for k,v in model.named_parameters():
#     if 'ptm_encoder.' not in k:
#         print('{}:{}'.format(k,v.size()))
# print('*' * 20, 'param not ptm', '*' * 20)

# model_param = list(model.parameters())
# ptm_param = list(model.ptm_encoder.parameters())
# ptm_param_id = list(map(id,ptm_param))
# for name, param in model.named_parameters():
#     if ''

# non_ptm_param = list(filter(lambda x:id(x) not in ptm_param_id,model_param))
model.ptm_encoder.unk_label_embedding = nn.Parameter(torch.zeros(size=[model.args.cls_hidden*model.args.cls_ff]),
                                                     requires_grad=False if args.unk_label_embed_zero else True)
if 'label' in args.train_mode:
    if args.label_embed_use_last_cls:
        model.ptm_encoder.label_embedding_weight = model.classfiers[-1].transformer_cls.layers[0].ffn[-1].weight
    else:
        # model.ptm_encoder.label_embedding = nn.Parameter(model.classfiers[-1].transformer_cls.layers[0].ffn[-1].weight)
        model.ptm_encoder.label_embedding_weight = nn.Parameter(model.classfiers[-1].transformer_cls.layers[0].ffn[-1].weight)
    model.ptm_encoder.w_label_embedding = nn.Linear(model.args.cls_hidden*model.args.cls_ff,model.args.cls_hidden,bias=False)


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


assert len(model.classfiers) == 1
import torch.nn as nn
from V0.modeling_bert_ee import TENER_classifier,Pseudo_Classifier
target_size = len(bundle.vocabs['target'])
print('model after_bert:{}'.format(model_args.after_bert))
if model_args.after_bert == 'tener':
    classfiers = [
        TENER_classifier(model_args, model.classfiers[0].input_size, target_size) for _ in range(11)]
    classfiers.append(model.classfiers[0])
else:
    classfiers = [
        Pseudo_Classifier(model_args, model.classfiers[0].input_size, target_size) for _ in range(11)]
    classfiers.append(model.classfiers[0])
model.classfiers = nn.ModuleList(classfiers)

for param in model.parameters():
    param.requires_grad = False

# for param in model.classfiers[:-1].parameters():
#     param.requires_grad = True

for cls in model.classfiers[:-1]:
    for param in cls.parameters():
        param.requires_grad = True



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
    raise NotImplementedError
    from utils import Unfreeze_Callback
    # model.ptm_encoder.requires_grad = False
    for k,v in model.ptm_encoder.named_parameters():
        v.requires_grad = False
    print('model param freeze!')
    # exit()
    callbacks.append(Unfreeze_Callback(model.ptm_encoder,fix_epoch_num=args.fix_ptm_epoch))
elif args.fix_ptm_step>0:
    raise NotImplementedError
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




# if args.early_stop_patience>0:
#     callbacks.append(EarlyStopCallback(args.early_stop_patience))

metrics = []
english_pos_dataset = ['ritter_pos','ark_twitter_pos']
# acc_metric = AccuracyMetric(pred='pred',target='target')
# acc_metric.set_metric_name('acc')
if args.train_mode == 'one_cls':
    f_metric = SpanFPreRecMetric(-1,bundle.vocabs['target'],'pred','target',encoding_type=args.encoding_type)
    f_metric.set_metric_name('l_12')
    metrics.append(f_metric)
    acc_metric = AccuracyMetric(-1,pred='pred',target='target')
    acc_metric.set_metric_name('acc_12')
    metrics.append(acc_metric)



elif 'joint' in args.train_mode:
    i = 10
    if model_args.dataset not in english_pos_dataset:
        f_metric = SpanFPreRecMetric(i, bundle.vocabs['target'], 'pred', 'target', encoding_type=args.encoding_type)
        f_metric.set_metric_name('l_{}'.format(i + 1))
        metrics.append(f_metric)

    acc_metric = AccuracyMetric(i, pred='pred', target='target')
    acc_metric.set_metric_name('acc_{}'.format(i + 1))
    metrics.append(acc_metric)

    for i in range(10):
        if model_args.dataset not in english_pos_dataset:

            f_metric = SpanFPreRecMetric(i, bundle.vocabs['target'], 'pred', 'target', encoding_type=args.encoding_type)
            f_metric.set_metric_name('l_{}'.format(i+1))
            metrics.append(f_metric)

        acc_metric = AccuracyMetric(i,pred='pred', target='target')
        acc_metric.set_metric_name('acc_{}'.format(i+1))
        metrics.append(acc_metric)

    for i in [11]:
        if model_args.dataset not in english_pos_dataset:

            f_metric = SpanFPreRecMetric(i, bundle.vocabs['target'], 'pred', 'target', encoding_type=args.encoding_type)
            f_metric.set_metric_name('l_{}'.format(i+1))
            metrics.append(f_metric)

        acc_metric = AccuracyMetric(i,pred='pred', target='target')
        acc_metric.set_metric_name('acc_{}'.format(i+1))
        metrics.append(acc_metric)

# metrics.append(acc_metric)
language_ = model_args.language_
args.use_bigram = model_args.use_bigram
args.ptm_pool_method = model_args.ptm_pool_method
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

if bundle.datasets.get('test'):
    fitlog_callback = FitlogCallback(data={'train':bundle.datasets['train'][:2000],'test':bundle.datasets['test']},log_loss_every=10)
else:
    fitlog_callback = FitlogCallback(data=bundle.datasets['train'][:2000],log_loss_every=10)

if args.train_part>0:
    callbacks.append(FitlogCallback())
else:
    callbacks.append(fitlog_callback)
# callbacks.append(FitlogCallback())
import torch.nn as nn

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


if args.if_save:
    from utils import get_peking_time
    exp_tag = get_peking_time()
    fitlog.add_other(str(exp_tag)[5:],'time_tag')
    exp_path = 'exps_further/{}'.format(exp_tag)
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


from fastNLP import SequentialSampler,BucketSampler
if args.sampler == 'bucket':
    sampler = BucketSampler()
elif args.sampler  == 'sequential':
    sampler = SequentialSampler()

from fastNLP import cache_results
from fastNLP.core.batch import DataSetIter
from fastNLP.core.trainer import _move_dict_value_to_device
import torch.nn as nn
from tqdm import tqdm
@cache_results(_cache_fp='cache/{}/{}'.format(args.exp_path,'epoch_{}'.format(args.ckpt_epoch) if args.ckpt_epoch>0 else 'step_{}'.format(args.ckpt_step)),
               _refresh=False)
def get_pred(bundle,model):
    pred_list = []
    loss_list = []
    test_batch_size = 16
    model.eval()
    model = model.to(device)
    data_iterator = DataSetIter(dataset=bundle.datasets['train'], batch_size=test_batch_size, sampler=SequentialSampler(),
                num_workers=0, drop_last=False)
    print('len_data_iterator:{}'.format(len(data_iterator)))
    print('len_train:{}'.format(len(bundle.datasets['train'])))
    # exit()
    # for batch_x,batch_y in

    loss_func = nn.CrossEntropyLoss(reduction='none',ignore_index=-100)
    with torch.no_grad():
        pbar = tqdm(total=len(data_iterator))
        for batch_x, batch_y in data_iterator:
            now_batch_pred_list = []
            now_batch_loss_list = []
            indices = data_iterator.get_batch_indices()
            _move_dict_value_to_device(batch_x, batch_y, device=device)
            prediction_dict = model(**batch_x)
            prediction = prediction_dict['pred']
            prediction = torch.transpose(prediction,0,1)
            # print('prediction:{}'.format(prediction.size()))
            target = batch_y['target']
            seq_len = batch_x['seq_len']
            batch_size = target.size(0)

            # now_batch_pred_list = prediction.tolist()
            for i,p_12 in enumerate(prediction):
                # print('p_12:{}'.format(p_12.size()))
                now_batch_pred_list.append(p_12[:,:seq_len[i]].tolist())

            target_unsq = target.unsqueeze(0).expand(12,*target.size()).transpose(0,1)
            # print('target_unsq:{}'.format(target_unsq.size()))
            target_flat = target_unsq.flatten(0,2)
            prediction_flat = prediction.flatten(0,2)
            loss_flat = loss_func(prediction_flat,target_flat)
            loss = loss_flat.view(batch_size,12,torch.max(seq_len).item())
            # print('loss:{}'.format(loss.size()))

            for i,l_12 in enumerate(loss):
                # print('p_12:{}'.format(p_12.size()))
                now_batch_loss_list.append(l_12[:,:seq_len[i]].tolist())

            pred_list.extend(now_batch_pred_list)
            loss_list.extend(now_batch_loss_list)
            pbar.update(1)

    # print('pred_list:{}'.format(len(pred_list)))
    # print('loss_list:{}'.format(len(loss_list)))
            # exit()
    return pred_list,loss_list


# pred_list,loss_list = get_pred(bundle,model)


model.loss_func = nn.CrossEntropyLoss(ignore_index=-100,reduction='none')

# print('pred_list:{}'.format(len(pred_list)))
# print('loss_list:{}'.format(len(loss_list)))

# print('joint_weighted:{}'.format(model_args.joint_weighted))
# model_args.joint_weighted = 1
from fastNLP import Tester
from fastNLP import DataSet
# DataSet.set_padder()
# DataSet.set_pad_val()
# DataSet.add_field()


    # bundle.add
    # bundle.set_input('')

from utils import get_constrain_matrix
if model_args.dataset not in english_pos_dataset:
    constrain_both = get_constrain_matrix(tag_vocab=bundle.vocabs['target'],encoding_type=args.encoding_type,return_torch=True)
    constrain_both[0],constrain_both[1] = constrain_both[0].to(device), constrain_both[1].to(device)
    model.ptm_encoder.constrain_both = constrain_both

from tqdm import tqdm
# pbar = tqdm(total=1000)
# token_exit_layers = []
from utils import simulate_ee
# for ins in bundle.datasets['train'][:1000]:
#     old_pred = ins['old_pred']
#     ee_mode = 't_level_win_ee'
#     win_size = 5
#     threshold = 0.15
#     t_level_t = 0.2
#     device = device
#     token_exit_layers_tmp = simulate_ee(old_pred,ee_mode,win_size,threshold,t_level_t,device,constrain_both)
#     token_exit_layers.extend(token_exit_layers_tmp)
#     pbar.update(1)

# token_exit_layers_num = [0]*12
# for l in token_exit_layers:
#     token_exit_layers_num[l]+=1

# print('token_exit_layers_num:{}'.format(token_exit_layers_num))
# token_exit_layers_num_sum = sum(token_exit_layers_num)
# token_exit_layers_num = list(map(lambda x:x/token_exit_layers_num_sum,token_exit_layers_num))
# print('token_exit_layers_num:{}'.format(token_exit_layers_num))

if args.test_joint:
    tester = Tester(bundle.datasets['test'],model,metrics,16,device=device,verbose=True)
    tester.test()

# model_args.train_mode =
if args.train_mode == 'joint_sample_copy':
    from utils import Sample_Stop_Update_Callback
    sample_stop_update_callback = Sample_Stop_Update_Callback('t_level_win_ee',args.sandwich_small,args.sandwich_full,device,constrain_both,12,args.true_copy,
                                                              min_win_size=args.min_win_size,
                                                              max_win_size=args.max_win_size,
                                                              min_threshold=args.min_threshold,
                                                              max_threshold=args.max_threshold,
                                                              min_t_level_t=args.min_t_level_t,
                                                              max_t_level_t=args.max_t_level_t)
    callbacks.append(sample_stop_update_callback)
    # bundle.datasets['train'].set_input('should_exit_wp_padded_tensor')
    # bundle.datasets['train'].set_input('should_exit_word_padded_tensor')


# bundle.datasets['train'] = bundle.datasets['train'][:100]

# if args.flooding:
#     bundle.datasets['train'].add_field('flooding_bias',loss_list,)
    # bundle.datasets['train'].set_pad_val('flooding_bias',-1)
# bundle.datasets['train'].add_field('old_pred',pred_list,padder=None)

if args.train_part>0:
    bundle.datasets['train'] = bundle.datasets['train'][:args.train_part]

bundle.datasets['train'].set_input('raw_words')
bundle.datasets['train'].set_padder('raw_words',None)

bundle.datasets['train'].set_input('word_piece_num')
bundle.datasets['train'].set_padder('word_piece_num',None)



for k,v in bundle.datasets.items():
    v.set_input('words','seq_len','target','word_pieces','bert_attention_mask','word_piece_seq_len')
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

# if args.flooding:
#     print('I set flooding into input')
#     bundle.datasets['train'].set_input('flooding_bias')
#     bundle.datasets['train'].set_pad_val('flooding_bias', -1)

# bundle.datasets['train'].set_input('old_pred')
# bundle.datasets['train'].set_padder('old_pred',None)


model_args.train_mode = args.train_mode

# print(bundle.datasets['train'])
# exit()


trainer = Trainer(bundle.datasets['train'],model,optimizer,LossInForward(),args.batch_size,sampler,
                  n_epochs=args.epoch,dev_data=bundle.datasets['dev'],metrics=metrics,metric_key='f' if model_args.dataset not in english_pos_dataset else None,
                  callbacks=callbacks,test_use_tqdm=False,device=device,check_code_level=-1)



from tmp_fastnlp_module import MyDatasetIter
trainer.data_iterator = MyDatasetIter(dataset=bundle.datasets['train'], batch_size=trainer.batch_size, sampler=sampler,
                                             num_workers=0, drop_last=False)
trainer.train()

fitlog.finish()

