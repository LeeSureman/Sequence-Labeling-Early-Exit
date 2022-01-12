import sys
sys.path.append('..')
# from fastNLP import Tester
from tmp_fastnlp_module import MyTester as Tester
import argparse
import torch
from tmp_fastnlp_module import MySpanFPreRecMetric as SpanFPreRecMetric,MyAccuracyMetric as AccuracyMetric
# from V3.modeling_bert_ee import BERT_SeqLabel_EE
from cal_custom_flops import *
import fitlog
#ontonotes 200 897920,
# vanilla_ops_dict = {
#                     'ontonotes_cn': 18707069.233152,
#                     'msra_ner': 15622902.531072,
#                     'weibo': 1330401.604608,
#                     'clue_ner': 2402048.093184,
#                     'e_com': 1287559.839744,
#                     'ctb5_pos': 1250503.022592,
#                     'ud1_pos': 1740053.643264,
#                     'ud2_pos': 1746779.738112,
#                     'ctb5_seg': 1243539.505152,
#                     'ud_seg':1735097.573376,
#                     'twitter_ner':6916292.130816,
#                     'ritter_pos':284480.73216,
#                     'ark_twitter_pos':1037361.9456,
#                     'conll':6106458.498048
#                     }
vanilla_ops_dict = {'bert-base-cased_ontonotes_cn':-1}

performance_dict = {'01_22_17_21_40.402874-10':80.2206, #ontonotes_cn
                    '01_22_21_15_52.861970-16':68.0143, #weibo
                    '01_22_20_26_17.420957-14':79.7897, #clue_ner
                    '01_22_20_26_20.172354-15':81.3406, #e_com
                    '01_26_20_43_11.186118-43':91.6689, #conll
                    '01_26_20_43_11.381835-43':91.7204, #conll
                    '01_24_20_28_13.777275-39':78.9695, #twitter_ner
                    '01_26_21_24_20.518714-58':78.9038, #twitter_ner
                    '01_26_21_24_19.238958-54':79.0043, #twitter_ner
                    '01_23_12_26_19.545890-22':96.7071, #ctb_pos
                    '01_26_21_53_16.183535-47':95.3938, #ud2_pos
                    '01_23_14_40_49.399237-10':98.7722, #ctb_seg
                    '01_22_21_15_53.593049-14':98.1723, #ud_seg
                    '01_26_09_33_18.590000-53':92.2408,  #arrk_twitter_pos

                    '02_01_21_39_23.643335-27':91.6932,
                    '01_24_12_27_03.563835-12':91.6974


                    }

fitlog.set_log_dir('logs_test_sentee')
# fitlog.set_log_dir('logs_test')

parser = argparse.ArgumentParser()
# parser.add_argument('--exp_path',default='2021_01_11_16_52_07.333694')
# parser.add_argument('--exp_path',default='2021_01_14_01_17_33.051602')
# parser.add_argument('--exp_path',default='2021_01_14_01_19_30.614438')
# parser.add_argument('--exp_path',default='2021_01_11_13_40_12.648499')
# parser.add_argument('--exp_path',default='2021_01_15_15_06_20.479439')
# parser.add_argument('--exp_path',default='01_18_18_13_00.617075')
# parser.add_argument('--exp_path',default='01_16_20_22_17.823294')
parser.add_argument('--msg')
parser.add_argument('--exp_path_ckpt_epoch',default=None)

parser.add_argument('--exp_path',default='01_22_17_21_40.402874')


parser.add_argument('--ckpt_epoch',type=int,default=-1)
parser.add_argument('--device',default='0')
parser.add_argument('--criterion',choices=['entropy','max_p'],default='entropy')
parser.add_argument('--sentence_ee_pool',choices=['avg','max'],default='max')
parser.add_argument('--threshold',type=float,default=0.1)
parser.add_argument('--max_threshold',type=float,default=100)
parser.add_argument('--test_mode',choices=['one_cls','joint','s_level_ee','t_level_ee','t_level_ee_copy',
                                           't_level_win_ee','t_level_win_ee_copy_pseudo','t_level_win_ee_copy_pseudo_2','t_level_win_ee_copy',
                                           't_level_win_ee_copy_2','t_level_win_ee_copy_label'],
                    default='t_level_win_ee_copy')
parser.add_argument('--win_size',type=int,default=5)
parser.add_argument('--test_part',type=float,default=-1)
parser.add_argument('--use_constrain',type=int,default=0)
parser.add_argument('--t_level_t',default='-1')
parser.add_argument('--test_dataset',default='test',choices=['train','dev','test'])
parser.add_argument('--true_copy',type=int,default=1)
parser.add_argument('--true_label',type=int,default=1)
parser.add_argument('--test_joint',type=int,default=0)
parser.add_argument('--test_joint_flops',type=int,default=0)
parser.add_argument('--further',type=int,default=1)
parser.add_argument('--keep_norm_same',type=int,default=-1)
parser.add_argument('--record_flops',type=int,default=1)
parser.add_argument('--local_aware_1',type=int,default=0)#标签迁移有冲突的话就都不退
parser.add_argument('--local_aware_2',type=int,default=0)#最后做一个后处理
parser.add_argument('--copy_wordpiece',choices=['first','all'],default=-1)

parser.add_argument('--max_inference_layer',type=int,default=12)
# parser.add_argument('--t_level_t',default='0,0,0,0,0,0,0.01,0.01,0.01,0.01,0.01,0.01')
# parser.add_argument('--t_level_t',default='0,0,0,0,0,0,0.02,0.02,0.02,0.02,0.02,0.02')
# parser.add_argument('--t_level_t',default='0,0,0,0,0,0,0.02,0.02,0.03,0.04,0.05,0.05')
# parser.add_argument('--t_level_t',default='0,0,0,0,0,0,0.1,0.1,0.1,0.1,0.1,0.1')
# parser.add_argument('--t_level_t',default='0,0,0,0,0,0,0.1,0.1,0.1,0.15,0.15,0.15')
# parser.add_argument('--t_level_t',default='0,0,0,0,0,0,0.15,0.15,0.15,0.15,0.15,0.15')
# parser.add_argument('--t_level_t',default='0,0,0,0,0,0,0.2,0.2,.2,0.2,0.2,0.2')








english_pos_dataset = ['ritter_pos','ark_twitter_pos']
args = parser.parse_args()

if args.exp_path_ckpt_epoch is not None:
    tmp_split = args.exp_path_ckpt_epoch.split('-')
    assert len(tmp_split) == 2
    args.exp_path = tmp_split[0]
    args.ckpt_epoch = int(tmp_split[1])

fitlog.add_hyper(args)
if args.t_level_t == '-1' and (args.use_constrain):
    exit()

if args.exp_path[:5] == '2021_':
    args.exp_path = args.exp_path[5:]
else:
    # args.exp_path = '2021_{}'.format(args.exp_path)
    pass
# fitlog.add_hyper('dataset')

if ',' not in args.t_level_t:
    args.t_level_t = [-1]*6 + [float(args.t_level_t)]*6
else:
    if len(args.t_level_t)>0:
        t_level_t = args.t_level_t.split(',')
        assert len(t_level_t) == 12
        t_level_t = list(map(float,t_level_t))
        args.t_level_t = t_level_t
    else:
        args.t_level_t = [-1]*12


if args.test_mode == 't_level_win_ee' or args.test_mode == 't_level_win_ee_copy_pseudo' or 't_level_win_ee_copy_pseudo' in args.test_mode\
        or 't_level_win_ee_copy' in args.test_mode:
    args.threshold = [args.threshold]*6+[args.threshold*2]*6
    if args.max_threshold>0:
        args.threshold = list(map(lambda x:min(args.max_threshold,x),args.threshold))

if args.device not in ['cpu','all']:
    assert isinstance(args.device,int) or args.device.isdigit()
    device = torch.device('cuda:{}'.format(args.device))
elif args.device == 'cpu':
    device = torch.device('cpu')
elif args.device == 'all':
    device = [i for i in range(torch.cuda.device_count())]
else:
    raise NotImplementedError



model_path = 'exps{}/2021_{}/model/epoch_{}.pkl'.format('_further' if args.further else '',args.exp_path,args.ckpt_epoch)
bundle_path = 'exps{}/2021_{}/bundle.pkl'.format('_further' if args.further else '',args.exp_path)
args_path = 'exps{}/2021_{}/args.pkl'.format('_further' if args.further else '',args.exp_path)
main_args = args
args = torch.load(open(args_path,'rb'))



bundle = torch.load(open(bundle_path,'rb'))
print('target vocab size:{}'.format(len(bundle.vocabs['target'])))
print(len(bundle.vocabs['target']))
# exit()
# from utils import get_constrain_matrix
# m1,m2 = get_constrain_matrix(bundle.vocabs['target'],args.encoding_type)
# print('m1:\n{}'.format(m1))
# print('m2:\n{}'.format(m2))
#
# exit()




model = torch.load(open(model_path,'rb'),map_location=torch.device('cpu'))
from V0.modeling_albert_ee import AlbertAttention_EE_pseudo_counter
if 'albert' in model.args.ptm_name:
    model.ptm_encoder.encoder.albert_layer_groups[0].albert_layers[0].attention.counter = AlbertAttention_EE_pseudo_counter()
model.ptm_encoder.config.max_inference_layer = main_args.max_inference_layer
# print('model train_mode:{}'.format(model.args.train_mode))
# print('model test_mode:{}'.format(model.args.test_mode))
# print('model keep norm same:{}'.format(model.args.keep_norm_same))
# exit()
# bundle = torch.load(open(bundle_path,'rb'))
# args = torch.load(open(args_path,'rb'))
# args = torch.load(open(args_path,'rb'))

if main_args.test_mode not in ['joint','one_cls']:
    fitlog.add_hyper(model.args.dataset,'dataset')

fitlog.add_hyper(model.args.ptm_name,'ptm_name')
if model.args.dataset not in english_pos_dataset:
    from utils import get_constrain_matrix
    # if main_args.use_constrain:
    constrain_both = get_constrain_matrix(tag_vocab=bundle.vocabs['target'],encoding_type=args.encoding_type,return_torch=True)
    constrain_both[0],constrain_both[1] = constrain_both[0].to(device), constrain_both[1].to(device)
    model.ptm_encoder.constrain_both = constrain_both

args.test_mode = main_args.test_mode





# for a,b in bundle.vocabs['target']:
#     print(a,b)
# exit()




model.ptm_encoder.exit_layer_num = [0]*12
model.ptm_encoder.exit_layer_num_token = [0] * 12
model.ptm_encoder.inference_token_num = 0
model.ptm_encoder.every_ins_token_exit_layers = []

model.layer_exit_pred = []
for i in range(model.ptm_config.num_hidden_layers):
    model.layer_exit_pred.append([])

label_num = {}
for ins in bundle.datasets['test']:
    targets = ins['target']
    for target in targets:
        target = bundle.vocabs['target'].to_word(target)
        if target in label_num:
            label_num[target] += 1
        else:
            label_num[target] = 1

print(label_num)
label_num_sum = 0
for k,v in label_num.items():
    label_num_sum+=v

for k,v in label_num.items():
    label_num[k]/=label_num_sum
print(label_num)


for k,v in main_args.__dict__.items():
    print('{}:{}'.format(k,v))

print('dataset:{}'.format(model.args.dataset))



if main_args.test_part < 0:
    test_dataset = bundle.datasets[main_args.test_dataset]
elif main_args.test_part < 1:
    main_args.test_part = main_args.test_part * len(bundle.datasets[main_args.test_dataset])
    test_dataset = bundle.datasets[main_args.test_dataset][:int(main_args.test_part)]
else:
    # main_args.test_part = main_args.test_part * len(bundle.datasets['test'])
    test_dataset = bundle.datasets[main_args.test_dataset][:int(main_args.test_part)]

for cls in model.classfiers:
    cls.target_size = len(bundle.vocabs['target'])

from V0.modeling_bert_ee import TENER_classifier
from transformers.models.bert.modeling_bert import BertAttention
# from V0.modeling_albert_ee import AlbertAttention_EE_pseudo_counter
# custom_ops = {TENER_classifier:count_tener}
custom_ops = {}
custom_ops[BertAttention] = count_bert_self_att_called_by_bertlayer_ee
custom_ops[AlbertAttention_EE_pseudo_counter] = count_albert_self_att_called_by_bertlayer_ee
# custom_ops[TENER_classifier] = count_tener

english_pos_dataset = ['ritter_pos','ark_twitter_pos']

from fastNLP import cache_results
@cache_results(_cache_fp='cache/tmp_vanilla_model_f',_refresh=False)
def get_vanilla_model_f():

    model.args.test_mode = 'one_cls'
    metrics = []
    if model.args.dataset not in english_pos_dataset:
        f_metric = SpanFPreRecMetric(-1,bundle.vocabs['target'],'pred','target',encoding_type=args.encoding_type)
        f_metric.set_metric_name('l_12')
        metrics.append(f_metric)
    else:
        acc_metric = AccuracyMetric(-1,pred='pred',target='target')
        acc_metric.set_metric_name('acc_12')
        metrics.append(acc_metric)

    joint_tester = Tester(test_dataset,model,metrics,batch_size=1 if main_args.test_joint_flops else 16,device=device,use_tqdm=True,verbose=True,
                    record_flops=False,custom_ops=custom_ops)
    result = joint_tester.test()
    print('joint mode test result')
    print(result)
    # print(abs(result['l_12']['f']*100-performance_dict['{}-{}'.format(main_args.exp_path,main_args.ckpt_epoch)]))
    # if abs(result['l_12']['f']*100-performance_dict['{}-{}'.format(main_args.exp_path,main_args.ckpt_epoch)])<1e-3:
    #     fitlog.add_other(1,'joint_f_correct')
    # else:
    #     fitlog.add_other(0, 'joint_f_correct')
    # if main_args.test_joint_flops:
    #     print('total model flops:{}'.format(joint_tester.total_flops/(1000**2)))

    model.ptm_encoder.reset_stats()
    # print(result)
    # print(result['acc_12']['acc'] * 100)
    if model.args.dataset not in english_pos_dataset:
        return result['l_12']['f'] * 100
    else:
        return result['acc_12']['acc'] * 100
    # return result['l_12']['f']*100

backbone_performance = get_vanilla_model_f(_cache_fp='cache/vanilla_model_f_{}_{}'.format(main_args.exp_path,main_args.ckpt_epoch))
performance_dict['{}-{}'.format(main_args.exp_path,main_args.ckpt_epoch)] = backbone_performance

from tmp_thop import myprofile
@cache_results(_cache_fp='cache/tmp_model_flops',_refresh=False)
def get_model_flops():
    # tester =
    # flops, params = myprofile(model=model,/ inputs=x, verbose=False, custom_ops=self.custom_ops)
    metrics = []
    model.args.test_mode = 'one_cls'
    tester = Tester(test_dataset, model, metrics, batch_size=1, device=device, use_tqdm=True, verbose=True,
                    record_flops=main_args.record_flops, custom_ops=custom_ops)
    tester.test()
    model.args.test_mode = main_args.test_mode
    return tester.total_flops/(1000**2)

backbone_flops = get_model_flops(_cache_fp='cache/model_flops_{}_{}'.format(model.args.dataset,model.args.ptm_name),_refresh=False)
vanilla_ops_dict['{}_{}'.format(model.args.ptm_name,model.args.dataset)] = backbone_flops

if main_args.test_mode != 'joint' and main_args.test_joint:
    model.args.test_mode = 'joint'
    metrics = []

    for i in range(12):
        if model.args.dataset not in english_pos_dataset:
            f_metric = SpanFPreRecMetric(i, bundle.vocabs['target'], 'pred', 'target', encoding_type=args.encoding_type)
            f_metric.set_metric_name('l_{}'.format(i + 1))
            metrics.append(f_metric)

        acc_metric = AccuracyMetric(i, pred='pred', target='target')
        acc_metric.set_metric_name('acc_{}'.format(i + 1))
        metrics.append(acc_metric)

    joint_tester = Tester(test_dataset,model,metrics,batch_size=1 if main_args.test_joint_flops else 16,device=device,use_tqdm=True,verbose=True,
                    record_flops=main_args.test_joint_flops,custom_ops=custom_ops)
    result = joint_tester.test()
    print('joint mode test result')
    print(result)
    print(abs(result['l_12']['f']*100-performance_dict['{}-{}'.format(main_args.exp_path,main_args.ckpt_epoch)]))
    if abs(result['l_12']['f']*100-performance_dict['{}-{}'.format(main_args.exp_path,main_args.ckpt_epoch)])<1e-3:
        fitlog.add_other(1,'joint_f_correct')
    else:
        fitlog.add_other(0, 'joint_f_correct')
    if main_args.test_joint_flops:
        print('total model flops:{}'.format(joint_tester.total_flops/(1000**2)))
    model.ptm_encoder.reset_stats()

# fitlog.add_best_metric(-1, 'f_ee')
# exit()

model.ptm_encoder.reset_stats()
model.args.test_mode = main_args.test_mode
model.args.criterion = main_args.criterion
model.args.sentence_ee_pool = main_args.sentence_ee_pool
model.args.threshold = main_args.threshold
model.args.win_size = main_args.win_size
model.args.use_constrain = main_args.use_constrain
model.args.t_level_t = main_args.t_level_t
model.args.true_copy = main_args.true_copy
model.args.local_aware_1 = main_args.local_aware_1
model.args.local_aware_2 = main_args.local_aware_2
model.args.true_label = main_args.true_label

if main_args.copy_wordpiece !=-1:
    model.args.copy_wordpiece = main_args.copy_wordpiece
else:
    if hasattr(model.args,'copy_wordpiece'):
        pass
    else:
        model.args.copy_wordpiece = 'all'

if main_args.keep_norm_same >0:
    model.args.keep_norm_same = main_args.keep_norm_same

# if model.args.language_ == 'en':
#     print('test copy doesnt support en, exit!')
#     raise NotImplementedError

metrics = []
if args.test_mode == 'one_cls':
    if model.args.dataset not in english_pos_dataset:
        f_metric = SpanFPreRecMetric(-1,bundle.vocabs['target'],'pred','target',encoding_type=args.encoding_type)
        f_metric.set_metric_name('l_12')
        metrics.append(f_metric)
    acc_metric = AccuracyMetric(-1,pred='pred',target='target')
    acc_metric.set_metric_name('acc_12')
    metrics.append(acc_metric)
elif args.test_mode == 'joint':
    for i in range(12):
        if model.args.dataset not in english_pos_dataset:
            f_metric = SpanFPreRecMetric(i, bundle.vocabs['target'], 'pred', 'target', encoding_type=args.encoding_type)
            f_metric.set_metric_name('l_{}'.format(i+1))
            metrics.append(f_metric)

        acc_metric = AccuracyMetric(i,pred='pred', target='target')
        acc_metric.set_metric_name('acc_{}'.format(i+1))
        metrics.append(acc_metric)
elif args.test_mode == 's_level_ee':
    if model.args.dataset not in english_pos_dataset:
        f_metric = SpanFPreRecMetric(-1,bundle.vocabs['target'],'pred','target',encoding_type=args.encoding_type)
        f_metric.set_metric_name('l_ee')
        metrics.append(f_metric)
    acc_metric = AccuracyMetric(-1,pred='pred',target='target')
    acc_metric.set_metric_name('acc_ee')
    metrics.append(acc_metric)
elif args.test_mode == 't_level_ee':
    if model.args.dataset not in english_pos_dataset:
        f_metric = SpanFPreRecMetric(-1,bundle.vocabs['target'],'pred','target',encoding_type=args.encoding_type)
        f_metric.set_metric_name('l_ee')
        metrics.append(f_metric)
    acc_metric = AccuracyMetric(-1,pred='pred',target='target')
    acc_metric.set_metric_name('acc_ee')
    metrics.append(acc_metric)
elif args.test_mode == 't_level_win_ee':
    if model.args.dataset not in english_pos_dataset:
        f_metric = SpanFPreRecMetric(-1,bundle.vocabs['target'],'pred','target',encoding_type=args.encoding_type)
        f_metric.set_metric_name('l_ee')
        metrics.append(f_metric)
    acc_metric = AccuracyMetric(-1,pred='pred',target='target')
    acc_metric.set_metric_name('acc_ee')
    metrics.append(acc_metric)
elif args.test_mode == 't_level_win_ee_copy_pseudo':
    if model.args.dataset not in english_pos_dataset:
        f_metric = SpanFPreRecMetric(-1,bundle.vocabs['target'],'pred','target',encoding_type=args.encoding_type)
        f_metric.set_metric_name('l_ee')
        metrics.append(f_metric)
    acc_metric = AccuracyMetric(-1,pred='pred',target='target')
    acc_metric.set_metric_name('acc_ee')
    metrics.append(acc_metric)
elif args.test_mode == 't_level_win_ee_copy_pseudo_2':
    if model.args.dataset not in english_pos_dataset:
        f_metric = SpanFPreRecMetric(-1,bundle.vocabs['target'],'pred','target',encoding_type=args.encoding_type)
        f_metric.set_metric_name('l_ee')
        metrics.append(f_metric)
    acc_metric = AccuracyMetric(-1,pred='pred',target='target')
    acc_metric.set_metric_name('acc_ee')
    metrics.append(acc_metric)
elif args.test_mode == 't_level_win_ee_copy':
    if model.args.dataset not in english_pos_dataset:
        f_metric = SpanFPreRecMetric(-1,bundle.vocabs['target'],'pred','target',encoding_type=args.encoding_type)
        f_metric.set_metric_name('l_ee')
        metrics.append(f_metric)
    acc_metric = AccuracyMetric(-1,pred='pred',target='target')
    acc_metric.set_metric_name('acc_ee')
    metrics.append(acc_metric)
elif args.test_mode == 't_level_win_ee_copy_2':
    if model.args.dataset not in english_pos_dataset:
        f_metric = SpanFPreRecMetric(-1,bundle.vocabs['target'],'pred','target',encoding_type=args.encoding_type)
        f_metric.set_metric_name('l_ee')
        metrics.append(f_metric)
    acc_metric = AccuracyMetric(-1,pred='pred',target='target')
    acc_metric.set_metric_name('acc_ee')
    metrics.append(acc_metric)
elif args.test_mode == 't_level_win_ee_copy_label':
    if model.args.dataset not in english_pos_dataset:
        f_metric = SpanFPreRecMetric(-1,bundle.vocabs['target'],'pred','target',encoding_type=args.encoding_type)
        f_metric.set_metric_name('l_ee')
        metrics.append(f_metric)
    acc_metric = AccuracyMetric(-1,pred='pred',target='target')
    acc_metric.set_metric_name('acc_ee')
    metrics.append(acc_metric)
else:
    raise NotImplementedError
# if model.args.dataset not in english_pos_dataset:
#     
#     nb_f_metric = SequentialSpanFMetric(bundle.vocabs['target'], 'pred', 'target',encoding_type=args.encoding_type)
#     nb_f_metric.set_metric_name('l12_nb')
#     metrics.append(nb_f_metric)
#     vit_metric = ViterbiSpanFMetric(bundle.vocabs['target'], 'pred', 'target',encoding_type=args.encoding_type)
#     vit_metric.set_metric_name('l12_vit')
#     metrics.append(vit_metric)
# nb_f_metric.set_metric_name('l12_nb')
model_ = model.ptm_encoder

from fastNLP.modules import allowed_transitions

if model.args.dataset not in english_pos_dataset:
    model_.allowed_transitions = set(allowed_transitions(bundle.vocabs['target'], include_start_end=False, encoding_type=model.args.encoding_type))

# print(model_.inference_layers_num)
# exit()
test_dataset.set_input('first_word_pieces_pos','word_piece_num')
tester = Tester(test_dataset,model,metrics,batch_size=1,device=device,use_tqdm=True,verbose=True,
                record_flops=main_args.record_flops,custom_ops=custom_ops)
result = tester.test()



print(result)
print('test_dataset size:{}'.format(len(test_dataset)))
# print(model.)
if main_args.test_mode not in ['joint','one_cls']:
    if model.args.dataset not in english_pos_dataset:

        model_flops = tester.total_flops / (1000 ** 2)
        if main_args.record_flops:
            print('model flops:{}'.format(model_flops))
            print('flops_speedup:{}'.format(vanilla_ops_dict['{}_{}'.format(model.args.ptm_name,model.args.dataset)] / model_flops))
            fitlog.add_other(model_flops, 'model_flops')
        fitlog.add_best_metric(backbone_performance, 'backbone_f')

        fitlog.add_best_metric(result['l_ee']['f']*100, 'f_ee')
        fitlog.add_best_metric(result['l_ee']['f']*100-performance_dict['{}-{}'.format(main_args.exp_path,main_args.ckpt_epoch)], 'f_ee_dif')
        fitlog.add_best_metric(result['l_ee']['pre']*100, 'pre_ee')
        fitlog.add_best_metric(result['l_ee']['rec']*100, 'rec_ee')

        # fitlog.add_best_metric(result['l12_nb']['f']*100, 'f_ee_nb')
        # fitlog.add_best_metric(result['l12_nb']['pre']*100, 'pre_ee_nb')
        # fitlog.add_best_metric(result['l12_nb']['rec']*100, 'rec_ee_nb')
        # fitlog.add_best_metric(result['l12_vit']['f']*100, 'f_ee_vit')
        # fitlog.add_best_metric(result['l12_vit']['pre']*100, 'pre_ee_vit')
        # fitlog.add_best_metric(result['l12_vit']['rec']*100, 'rec_ee_vit')
    else:
        fitlog.add_best_metric(backbone_performance, 'acc')
        fitlog.add_best_metric(result['acc_ee']['acc']*100, 'acc_ee')
        fitlog.add_best_metric(result['acc_ee']['acc']*100-performance_dict['{}-{}'.format(main_args.exp_path,main_args.ckpt_epoch)], 'acc_ee_dif')


if main_args.test_mode == 's_level_ee':
    print('ins num:{}'.format(model_.inference_instances_num * 12))
    print('layer num:{}'.format(model_.inference_layers_num))
    print('layer time speedup:{}'.format(model_.inference_layers_num/(model_.inference_instances_num*12)))

    exit_layer_num = model_.exit_layer_num
    all_go_through_layer_num = sum(exit_layer_num)
    # print('all_go_through_layer_num:{}'.format(all_go_through_layer_num))
    # print('model_.inference_layers_num:{}'.format(model_.inference_layers_num))
    # print('exit_layer_num:{}'.format(exit_layer_num))
    exit_layer_num = list(map(lambda x:x/all_go_through_layer_num,exit_layer_num))
    print('exit_layer_num:{}'.format(exit_layer_num))
    model_flops = tester.total_flops / (1000 ** 2)
    if main_args.record_flops:
        print('model flops:{}'.format(model_flops))
        print('flops_speedup:{}'.format(vanilla_ops_dict['{}_{}'.format(model.args.ptm_name,model.args.dataset)]/model_flops))
        fitlog.add_other(model_flops, 'model_flops')

    fitlog.add_other(vanilla_ops_dict['{}_{}'.format(model.args.ptm_name,model.args.dataset)]/model_flops,'flops_speedup')
    fitlog.add_other((model_.inference_instances_num*12)/model_.inference_layers_num,'layer_time_speedup')
    fitlog.add_to_line('exit_layer_num:{}'.format(exit_layer_num))




elif main_args.test_mode in ['t_level_ee','t_level_win_ee','t_level_win_ee_copy_pseudo','t_level_win_ee_copy_pseudo_2','t_level_win_ee_copy',
                             't_level_win_ee_copy_2','t_level_win_ee_copy_label']:

    exit_layer_num = model_.exit_layer_num
    all_go_through_layer_num = sum(exit_layer_num)
    print('exit_layer_num:{}'.format(exit_layer_num))
    exit_layer_num = list(map(lambda x:x/all_go_through_layer_num,exit_layer_num))
    print('exit_layer_num:{}'.format(exit_layer_num))

    print('all token num:{}'.format(model_.inference_token_num))
    print('exit_layer_num_token:{}'.format(model_.exit_layer_num_token))
    token_level_layer_num_sum = 0
    for i,token_num in enumerate(model_.exit_layer_num_token):
        token_level_layer_num_sum += ((i+1)*token_num)
    exit_layer_num_token = list(map(lambda x:round(x/model_.inference_token_num,6),model_.exit_layer_num_token))
    print('exit_layer_num_token:{}'.format(exit_layer_num_token))
    print('token time speedup:{}'.format(token_level_layer_num_sum/(model_.inference_token_num*12)))

    # exit_layer_num_sum = sum(exit_layer_num)
    # non_ee_layer_num = len(test_dataset)*12
    # print(exit_layer_num_sum)
    # print(non_ee_layer_num)
    # if main_args.record_flops:
    #     model_.inference_layers_num/=2
    print(model_.inference_layers_num)
    print(model_.inference_instances_num)
    # print(non_ee_layer_num)

    print('layer time speedup:{}'.format(model_.inference_layers_num/(model_.inference_instances_num*12)))

    model_flops = tester.total_flops / (1000 ** 2)
    if main_args.record_flops:
        print('model flops:{}'.format(model_flops))
        print('flops_speedup:{}'.format(vanilla_ops_dict['{}_{}'.format(model.args.ptm_name,model.args.dataset)]/model_flops))
        fitlog.add_other(model_flops, 'model_flops')

    fitlog.add_to_line('exit_layer_num:{}'.format(exit_layer_num))
    fitlog.add_to_line('exit_layer_num_token:{}'.format(exit_layer_num_token))

    fitlog.add_other(vanilla_ops_dict['{}_{}'.format(model.args.ptm_name,model.args.dataset)]/model_flops,'flops_speedup')
    fitlog.add_other((model_.inference_instances_num*12)/model_.inference_layers_num,'layer_time_speedup')
    fitlog.add_other((model_.inference_token_num*12)/token_level_layer_num_sum,'token_time_speedup')

    # print(model.layer_exit_pred)



    # label_to_exit_layer = []
    # for i in range(len(bundle.vocabs['target'])):
    #     label_to_exit_layer.append([0]*12)
    #
    # layer_to_exit_label_num = []
    # for i in range(12):
    #     layer_to_exit_label_num.append([0]*len(bundle.vocabs['target']))
    #     for label in model.layer_exit_pred[i]:
    #         layer_to_exit_label_num[-1][label]+=1
    #         label_to_exit_layer[label][i]+=1
    #
    #
    # for i in range(12):
    #     print('layer:{}'.format(i))
    #     now_layer_exit_token_num_sum = sum(layer_to_exit_label_num[i])
    #     for label_name,label_id in bundle.vocabs['target']:
    #         print('{}:{}'.format(label_name,layer_to_exit_label_num[i][label_id]/now_layer_exit_token_num_sum),end=' * ')
    #     print(' ')
    #
    # # for i in range(len(bundle.vocabs['target'])):
    # #     print('for label')
    # for label_name, label_id in bundle.vocabs['target']:
    #     now_label_exit_layer_num_sum = sum(label_to_exit_layer[label_id])
    #     print('for {}:'.format(label_name))
    #     for i in range(12):
    #         print('{}:{}'.format(i,label_to_exit_layer[label_id][i]/now_label_exit_layer_num_sum),end=' * ')
    #     print(' ')

