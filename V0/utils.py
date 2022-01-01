import torch
# from fastNLP import Dataset
import torch.nn as nn





def get_crf_zero_init(label_size, include_start_end_trans=False, allowed_transitions=None,
                 initial_method=None):
    import torch.nn as nn
    from fastNLP.modules import ConditionalRandomField
    crf = ConditionalRandomField(label_size, include_start_end_trans)

    crf.trans_m = nn.Parameter(torch.zeros(size=[label_size, label_size], requires_grad=True))
    if crf.include_start_end_trans:
        crf.start_scores = nn.Parameter(torch.zeros(size=[label_size], requires_grad=True))
        crf.end_scores = nn.Parameter(torch.zeros(size=[label_size], requires_grad=True))
    return crf

def batch_index_select_through_seq(inp,seq_index):
    '''

    :param inp: [B, T, H] batch_size, time(seq_len), hidden_size
    :param index: [B]
    :return: [B,H]
    '''
    assert type(seq_index) is torch.Tensor

    batch_index = torch.arange(0,inp.size(0))
    seq_index = seq_index
    # print('batch_index:{}'.format(batch_index.size()))
    # print('seq_index:{}'.format(seq_index.size()))
    result = inp[batch_index,seq_index]

    return result.squeeze(dim=1)

def batch_span_select_through_seq(inp,head_index,tail_index):
    assert type(head_index) is torch.Tensor

    max_seq_len = inp.size(1)

    batch_index = torch.arange(0, inp.size(0))
    batch_size = inp.size(0)

    # print('batch_size:{}'.format(batch_size))
    # print('head_index:{}'.format(head_index.size()))

    max_span_len = torch.max(tail_index - head_index + 1)

    # print('max_span_len:{}'.format(max_span_len))

    span_index = torch.arange(0, max_span_len).to(head_index)
    span_index = span_index.unsqueeze(0)
    span_index = span_index.expand([batch_size,*span_index.size()[1:]]).clone()
    head_index = head_index.unsqueeze(-1)

    span_index += head_index

    span_index = torch.clamp(span_index,-1,max_seq_len-1)

    batch_index = batch_index.unsqueeze(-1).expand([*batch_index.size(),max_span_len])
    # print('span_index:{}'.format(span_index))
    # print('span_index:{}'.format(span_index.size()))
    result = inp[batch_index,span_index]
    result = result.view([batch_size,max_span_len,-1])

    return result

def batch_sample_all_span_by_length(inp,span_len):
    '''

    :param inp: [B, T, H] batch, time, hidden_size
    :param length: span_len, int
    :return: [ batch, num, length, hidden ]
    '''
    # print(inp)
    # print(inp.size())
    inp = inp.permute(0, 2, 1)
    # print(inp.size())
    inp = inp.unsqueeze(dim=2)
    # print(inp.size())

    result = torch.nn.functional.unfold(inp, (1,span_len))
    inp_unf = result
    # print('after infold:\n{} \n{}'.format(result.size(),result))

    inp_unf = inp_unf.view(inp.size(0), inp.size(1),span_len, inp.size(3) - span_len + 1)
    inp_unf = inp_unf.permute(0, 3, 2, 1)

    return inp_unf.contiguous()

def dict_output(dict_):
    result = ''
    for k,v in dict_.items():
        result = result + '({}:{:.3}) '.format(k,v)

    return result[:-1]

def shuffle_dataset(dataset,shuffle_seed=-1):
    from fastNLP import DataSet
    import random
    result = DataSet()
    ins_list = []
    for ins in dataset:
        ins_list.append(ins)
    if shuffle_seed>0:
        random.seed(shuffle_seed)
    random.shuffle(ins_list)
    for ins in ins_list:
        result.append(ins)

    return result

def get_ptm_from_name(ptm_name,vocab,pool_method,word_dropout=0.01,**kwargs):
    from fastNLP.embeddings import RobertaEmbedding,ElmoEmbedding,BertEmbedding
    # from tmp_fastnlp_module import BertEmbedding
    if ptm_name[:4] == 'bert':
        ptm_encoder = BertEmbedding(vocab,
                                    model_dir_or_name=ptm_name[5:],
                                    pool_method=pool_method, word_dropout=word_dropout,
                                     **kwargs)
    elif ptm_name[:7] == 'roberta':
        ptm_encoder = RobertaEmbedding(vocab,
                                    model_dir_or_name=ptm_name[8:],
                                    pool_method=pool_method, word_dropout=word_dropout,
                                     **kwargs)
        # raise NotImplementedError
    elif ptm_name[:4] == 'elmo':
        raise NotImplementedError


    return ptm_encoder






# if __name__ == '__main__':
#
#     a = torch.zeros(size=[2,3,4])
#     for i in range(a.size(0)):
#         for j in range(a.size(1)):
#             for k in range(a.size(2)):
#                 a[i,j,k] = 10*(10*i+j)+k
#
#     head_index = torch.tensor([0,0])
#     tail_index = torch.tensor([2,1])
#
#     b = batch_span_select_through_seq(a,head_index,tail_index)
#     print(a)
#     print(b)
#
#
#     # a = torch.rand(size=[2,4,6])
#     # for i in range(a.size(0)):
#     #     for j in range(a.size(1)):
#     #         for k in range(a.size(2)):
#     #             a[i,j,k] = 10*(10*i+j)+k
#     #
#     # b = batch_sample_all_span_by_length(a,2)
#     # print(a)
#     # print(b)
#     # print(b.size())
from fastNLP import Callback
class Unfreeze_Callback(Callback):
    def __init__(self,bert_embedding,fix_epoch_num=-1,fix_step_num=-1):
        super().__init__()
        self.bert_embedding = bert_embedding
        self.fix_epoch_num = fix_epoch_num
        self.fix_step_num = fix_step_num
        self.type_ = None
        if self.fix_epoch_num > 0:
            self.type_ = 'epoch'
        elif self.fix_step_num > 0:
            self.type_ = 'step'
        else:
            print('give me epoch or step! (unfreeze_callback)')
            exit()
        # assert self.bert_embedding.requires_grad == False
        for k,v in self.bert_embedding.named_parameters():
            assert v.requires_grad == False

    def on_epoch_begin(self):
        if self.type_ == 'epoch':
            if self.epoch == self.fix_epoch_num+1:
                # self.bert_embedding.requires_grad = True
                for k, v in self.bert_embedding.named_parameters():
                    v.requires_grad = True
        # print('{}:{}'.format(self.epoch, self.bert_embedding.requires_grad))
        # print(self.bert_embedding.encoder.layer[11].attention.self.query.weight[:10, 10])

    def on_step_end(self):
        if self.type_ == 'step':
            if self.step == self.fix_step_num:
                for k, v in self.bert_embedding.named_parameters():
                    v.requires_grad = True


class MyDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        assert 0<=p<=1
        self.p = p

    def forward(self, x):
        if self.training and self.p>0.0001:
            # print('mydropout!')
            mask = torch.rand(x.size())
            # print(mask.device)
            mask = mask.to(x)
            # print(mask.device)
            mask = mask.lt(self.p)
            x = x.masked_fill(mask, 0)/(1-self.p)
        return x

def batch_index_select_yf(tensor, index):
#     """
#     tensor [B, T1, H]
#     index [B, T2]
#     return [B, T2, H]
#     """
    errors = (index >= tensor.size(1)).sum().item()
    assert errors == 0, errors
    batch_idx = torch.arange(0, tensor.size(0), device=tensor.device)
    batch_idx = batch_idx.view(-1, 1).expand_as(index)
    return tensor[batch_idx, index]


def get_bigrams(words):
    result = []
    for i,w in enumerate(words):
        if i!=len(words)-1:
            result.append(words[i]+words[i+1])
        else:
            result.append(words[i]+'<end>')

    return result

from fastNLP.core.metrics import _bmeso_tag_to_spans,_bio_tag_to_spans
def bmeso_to_bio(label_seq):
    spans = _bmeso_tag_to_spans(label_seq)

    bio_label_seq = ['O']*len(label_seq)

    for type_,(s,e) in spans:
        assert bio_label_seq[s] == 'O'
        bio_label_seq[s] = 'B-{}'.format(type_.upper())
        for i in range(s+1,e):
            if i>=e:
                break
            assert bio_label_seq[i] == 'O'
            bio_label_seq[i] = 'I-{}'.format(type_.upper())

    spans_got_by_bio = _bio_tag_to_spans(bio_label_seq)
    assert spans_got_by_bio == spans

    return bio_label_seq

from fastNLP import Vocabulary
def transform_bmeso_bundle_to_bio(bundle):
    # from fastNLP.io.data_bundle import DataBundle
    bmeso_target_vocab = bundle.vocabs['target']
    bundle.apply_field(lambda x:list(map(bmeso_target_vocab.to_word,x)),field_name='target',new_field_name='raw_target')
    bundle.apply_field(bmeso_to_bio,field_name='raw_target',new_field_name='raw_target')

    bio_target_vocab = Vocabulary(padding=None,unknown=None)
    bio_target_vocab.from_dataset(*bundle.datasets.values(),field_name='raw_target')
    bio_target_vocab.index_dataset(*bundle.datasets.values(),field_name='raw_target',new_field_name='target')

    bundle.vocabs['target'] = bio_target_vocab

    return bundle

def transform_bio_bundle_to_bioes(bundle):
    # from fastNLP.io.data_bundle import DataBundle
    from fastNLP.io.pipe.utils import iob2bioes
    bio_target_vocab = bundle.vocabs['target']
    bundle.apply_field(lambda x:list(map(bio_target_vocab.to_word,x)),field_name='target',new_field_name='raw_target')
    bundle.apply_field(iob2bioes,field_name='raw_target',new_field_name='raw_target')

    bio_target_vocab = Vocabulary(padding=None,unknown=None)
    bio_target_vocab.from_dataset(*bundle.datasets.values(),field_name='raw_target')
    bio_target_vocab.index_dataset(*bundle.datasets.values(),field_name='raw_target',new_field_name='target')

    bundle.vocabs['target'] = bio_target_vocab

    return bundle





def get_peking_time():
    import time
    import datetime
    import pytz

    tz = pytz.timezone('Asia/Shanghai')  # 东八区

    t = datetime.datetime.fromtimestamp(float(time.time()), pytz.timezone('Asia/Shanghai')).strftime('%Y_%m_%d_%H_%M_%S.%f')
    return t


def get_entropy(x):
    # x: torch.Tensor, logits BEFORE softmax
    exp_x = torch.exp(x)
    A = torch.sum(exp_x, dim=1)    # sum of exp(x_i)
    B = torch.sum(x*exp_x, dim=1)  # sum of x_i * exp(x_i)
    return torch.log(A) - B/A

import math
def get_uncertainty(x):
    # x: torch.Tensor, logits BEFORE softmax
    num_tags = x.size(-1)
    entropy_x = get_entropy(x)
    return entropy_x/math.log(num_tags)

def get_entropy_2(x,need_softmax=True):
    if need_softmax:
        x = nn.functional.softmax(x,dim=-1)
    x_logx = torch.log(x)*x
    return -torch.sum(x_logx,dim=-1)

import math
def get_uncertainty_2(x,need_softmax=True):
    # x: torch.Tensor, logits BEFORE softmax
    num_tags = x.size(-1)
    entropy_x = get_entropy_2(x,need_softmax=need_softmax)
    return entropy_x/math.log(num_tags)


def exit_layers_to_should_exit(exit_layers,num_layers=12):
    # [seq_len]
    if type(exit_layers) is list:
        exit_layers = torch.tensor(exit_layers)
    result_list = []

    for i in range(num_layers):
        # result = torch.stack()
        result_list.append(exit_layers>=i)

    result = torch.stack(result_list)

    return result.long()



import numpy as np
def sample_token_dropout(batch_size,seq_len,bert_num_layers,max_seq_len=None,pooling_win_size=-1):
    if type(seq_len) is list:
        seq_len = torch.tensor(seq_len)
    assert bert_num_layers == 12
    if not max_seq_len:
        max_seq_len = torch.max(seq_len).item()
    #0 全random
    #1 过所有（standard）
    #2 一部分第一二三层就出，剩余random
    #3 一部分全random，剩余过所有
    #4 一部分第一二三层就出，剩余过所有

    sample_exit_type = torch.randint(0,3,size=[batch_size]).tolist()
    # sample_exit_type = [2]
    result = torch.zeros(size=[12,batch_size,max_seq_len])
    layer_p_1 = [5,3,2,1,1,1,1,1,1,1,1,1]
    p_sum = sum(layer_p_1)
    layer_p_1 = list(map(lambda x:x/p_sum,layer_p_1))

    layer_p_2 = [0.765406, 0.076942, 0.044313, 0.02578, 0.022344, 0.010944, 0.007767, 0.00559, 0.00447, 0.001812, 0.00185, 0.032783]
    p_sum = sum(layer_p_2)
    layer_p_2 = list(map(lambda x:x/p_sum,layer_p_2))

    for i in range(batch_size):
        if sample_exit_type[i] == 0:

        # token_exit_layers = torch.randint(0, bert_num_layers, size=[seq_len[i]])
            token_exit_layers = np.random.choice(bert_num_layers,size=[seq_len[i]],replace=True,p=layer_p_1)
            if pooling_win_size>0:
                assert pooling_win_size%2==1
                token_exit_layers = nn.functional.max_pool1d(token_exit_layers.unsqueeze(0).unsqueeze(0).float(),
                                                             kernel_size=pooling_win_size,stride=1,padding=pooling_win_size//2).squeeze(0).squeeze(0)
            token_should_exit = exit_layers_to_should_exit(token_exit_layers)
            result[:,i,:seq_len[i]] = token_should_exit

        if sample_exit_type[i] == 1:
            result[:,i,:seq_len[i]] = torch.ones(size=[12,seq_len[i]])
        elif sample_exit_type[i] == 2:
            token_exit_layers = np.random.choice(bert_num_layers,size=[seq_len[i]],replace=True,p=layer_p_2)
            if pooling_win_size>0:
                assert pooling_win_size%2==1
                token_exit_layers = nn.functional.max_pool1d(token_exit_layers.unsqueeze(0).unsqueeze(0).float(),
                                                             kernel_size=pooling_win_size,stride=1,padding=pooling_win_size//2).squeeze(0).squeeze(0)
            token_should_exit = exit_layers_to_should_exit(token_exit_layers)
            result[:,i,:seq_len[i]] = token_should_exit

        # elif



    return result







from fastNLP.modules import allowed_transitions
def get_constrain_matrix(tag_vocab,encoding_type,return_torch=False):
    trans = allowed_transitions(tag_vocab, include_start_end=False, encoding_type=encoding_type)
    constrain = torch.full((len(tag_vocab), len(tag_vocab)), fill_value=-10000.0, dtype=torch.float)
    for from_tag_id, to_tag_id in trans:
        constrain[from_tag_id, to_tag_id] = 0
    if return_torch:
        return [constrain,torch.transpose(constrain,0,1)]
    constrain = constrain.numpy()
    return [constrain,np.transpose(constrain)]

import copy
def mask_logit_by_certain_pred_and_constrain(logits,pred,constrain_both):
    '''

    :param logit: torch tensor [1,seq_len,num_types]
    :param pred: list[ tensor [num_types] ]
    :param constrain_both: [num_types, num_types], [num_types, num_types]
    :return:
    '''
    logits = copy.deepcopy(logits)
    constrain,constrain_inverse = constrain_both
    assert logits.size(0) == 1
    seq_len = len(pred)
    for i,pred_logit in enumerate(pred):
        if pred_logit is not None:
            if i != 0:
                logits[0,i-1] += constrain_inverse[torch.argmax(pred_logit,dim=-1).item()]

            if i != seq_len-1:
                logits[0,i+1] += constrain[torch.argmax(pred_logit,dim=-1).item()]

    return logits


import random

@torch.no_grad()
def simulate_ee(old_pred,ee_mode,win_size,threshold,t_level_t,device,constrain_both,num_hidden_layers=12,criterion='entropy',max_threshold_2=1000):
    '''

    :param old_pred: seq_len, num_types
    :param ee_mode:
    :param win_size:
    :param threshold:
    :param t_level_t:
    :param device:
    :param constrain_both:
    :param num_hidden_layers:
    :param criterion:
    :return:
    '''
    # pass
    threshold = [threshold]*6 + [threshold*2] * 6
    threshold = list(map(lambda x:min(x,max_threshold_2),threshold))
    t_level_t = [0]*6 + [t_level_t]*6
    assert num_hidden_layers == 12
    assert ee_mode == 't_level_win_ee'
    seq_len = len(old_pred[0])
    target_size = len(old_pred[0][0])

    token_exit_layers = [num_hidden_layers-1]*seq_len

    tmp = random.randint(0,1000)

    # use_constrain = tmp%2
    use_constrain = 0
    if use_constrain:
        mask_tensor = torch.zeros(size=[1, seq_len, target_size],
                                  dtype=torch.float).to(device).float()

    old_pred = torch.tensor(old_pred,dtype=torch.float,device=device)
    pred = [None] * seq_len
    for i in range(num_hidden_layers):
        logits = old_pred[i]
        if use_constrain and i > 0:
            logits += mask_tensor
            # logits = mask_logit_by_certain_pred_and_constrain(logits,pred,self.constrain_both)
            # pass
        # print('logits:{}'.format(logits.size()))
        if (i + 1) == num_hidden_layers:
            # for j, _ in enumerate(pred):
            #     if pred[j] is None:
            #         pass
            #         self.exit_layer_num_token[i] += 1
            #         pred[j] = logits[0, j]
            #         exit_layer[j] = i
            # self.exit_layer_num[i] += 1
            # self.inference_layers_num += (i + 1)
            return token_exit_layers

        # all_finished = True
        if criterion == 'entropy':
            tmp_for_judging_ee_token = get_uncertainty(logits)
            # tmp_for_judging_ee_token =
            tmp_for_judging_ee_token_uns = tmp_for_judging_ee_token.unsqueeze(0).unsqueeze(0)
            tmp_for_judging_ee = nn.functional.max_pool1d(tmp_for_judging_ee_token_uns, kernel_size=win_size,
                                                          stride=1,
                                                          padding=win_size // 2).squeeze(0).squeeze(0)
            # print('tmp:{}'.format(tmp_for_judging_ee.size()))
            # exit()

            for j, uncertainty in enumerate(tmp_for_judging_ee):
                if pred[j] is None and (
                        uncertainty < threshold[i] or tmp_for_judging_ee_token[j] < t_level_t[i]):
                    # self.exit_layer_num_token[i] += 1
                    pred[j] = logits[j]
                    token_exit_layers[j] = i
                    if use_constrain:
                        if j != 0:
                            mask_tensor[0, j - 1] += constrain_both[1][torch.argmax(pred[j], dim=-1).item()]
                        if j != seq_len - 1:
                            mask_tensor[0, j + 1] += constrain_both[0][torch.argmax(pred[j], dim=-1).item()]
                    # exit_layer[j] = i
        elif criterion == 'max_p':
            raise NotImplementedError
            logits = nn.functional.softmax(logits, dim=-1)
            for j, p in enumerate(logits[0]):
                if pred[j] is None and p > self.args.threshold:
                    self.exit_layer_num_token[i] += 1
                    pred[j] = logits[0, j]
                    exit_layer[j] = i

        all_finished = True

        for p in pred:
            if p is None:
                all_finished = False
                break

        if all_finished:
            # self.exit_layer_num[i] += 1
            # self.inference_layers_num += (i + 1)
            return token_exit_layers


def inverse_pull_word_to_wordpiece(token_exit_layers,word_to_wordpiece_num,num_hidden_layers,copy_wordpiece):
    '''

    :param token_exit_layers: [seq_len]，每个位置表示这个word在哪层退
    :param word_to_wordpiece_num: [seq_len],每个位置表示这个word对应的wordpiece数量是多少
    :return:
    '''
    token_exit_layers_wp = [num_hidden_layers-1] # cls

    for i,l in enumerate(token_exit_layers):
        if copy_wordpiece == 'all':
            token_exit_layers_wp.extend([l]*word_to_wordpiece_num[i])
        elif copy_wordpiece == 'first':
            token_exit_layers_wp.append(l)
            token_exit_layers_wp.extend([num_hidden_layers-1]*(word_to_wordpiece_num[i]-1))

    token_exit_layers_wp.append(num_hidden_layers-1)
    return token_exit_layers_wp


import random
class Sample_Stop_Update_Callback(Callback):
    def __init__(self,ee_mode,sandwich_samll,sandwich_full,device,constrain_both,num_hidden_layers,true_copy,
                 min_win_size,max_win_size,min_threshold,max_threshold,max_threshold_2,min_t_level_t,max_t_level_t,copy_wordpiece,if_random):
        super().__init__()
        self.ee_mode = ee_mode
        self.win_sizes = [3,5,7,9,11,13,15]
        self.thresholds = [0.01,0.03,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
        self.t_level_ts = [-1,0.01,0.03,0.05,0.1,0.2,0.3]
        self.sandwich_small = sandwich_samll
        self.sandwich_full = sandwich_full
        self.device_ = device
        self.constrain_both = constrain_both
        self.num_hidden_layers = num_hidden_layers
        self.true_copy = true_copy

        self.min_win_size = min_win_size
        self.max_win_size = max_win_size
        self.min_t_level_t = min_t_level_t
        self.max_t_level_t = max_t_level_t
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.max_threshold_2 = max_threshold_2
        self.copy_wordpiece = copy_wordpiece

        self.if_random = if_random





        self.win_sizes = list(filter(lambda x:x<=max_win_size and x>=min_win_size,self.win_sizes))
        self.thresholds = list(filter(lambda x:x<=max_threshold and x>=min_threshold,self.thresholds))
        self.t_level_ts = list(filter(lambda x:x<=max_t_level_t and x>=min_t_level_t,self.t_level_ts))

        print('sample_copy_cvallback.win_sizes:{}'.format(self.win_sizes))
        print('sample_copy_cvallback.thresholds:{}'.format(self.thresholds))
        print('sample_copy_cvallback.t_level_ts:{}'.format(self.t_level_ts))


    def on_batch_begin(self, batch_x, batch_y, indices):
        if not self.true_copy:
            max_seq_len = batch_x['words'].size(1)
            max_wp_seq_len = batch_x['word_pieces'].size(1)
            batch_size = batch_x['words'].size(0)
            should_exit_word_padded_tensor = torch.ones(size=[batch_size,12,max_seq_len])
            should_exit_wp_padded_tensor = torch.ones(size=[batch_size,12,max_wp_seq_len])

            should_exit_word_padded_tensor = should_exit_word_padded_tensor.to(batch_x['words'])
            should_exit_wp_padded_tensor = should_exit_wp_padded_tensor.to(batch_x['words'])

            batch_x['should_exit_word_padded_tensor'] = should_exit_word_padded_tensor.bool()
            batch_x['should_exit_wp_padded_tensor'] = should_exit_wp_padded_tensor
            return None

        # print('on_batch_begin!')
        old_pred_batch = batch_x['old_pred']
        batch_size = len(old_pred_batch)

        now_batch_win_sizes = np.random.choice(self.win_sizes,size=[batch_size]).tolist()
        now_batch_thresholds = np.random.choice(self.thresholds,size=[batch_size]).tolist()
        now_batch_t_level_t = np.random.choice(self.t_level_ts,size=[batch_size]).tolist()

        # now_batch_win_sizes[0],now_batch_win_sizes[1] = self.win_sizes[0],self.win_sizes[0]
        # now_batch_thresholds[0],now_batch_thresholds[1] = self.thresholds[-1],self.thresholds[-1]
        # now_batch_

        for i in range(min(self.sandwich_small,batch_size)):
            tmp_sample = random.randint(0,1)
            if tmp_sample % 2:
                now_batch_win_sizes[i] = self.win_sizes[0]
            else:
                now_batch_thresholds[i] = self.thresholds[-1]
            now_batch_t_level_t[i] = self.t_level_ts[-1]

        token_exit_layers_batch = []
        for i,old_pred in enumerate(old_pred_batch):
            token_exit_layers = simulate_ee(old_pred,self.ee_mode,
                                                now_batch_win_sizes[i],now_batch_thresholds[i],now_batch_t_level_t[i],
                                                self.device_,constrain_both=self.constrain_both,max_threshold_2=self.max_threshold_2)
            if self.if_random:
                token_exit_layers = torch.randint(0,12,[len(token_exit_layers)])
            token_exit_layers_batch.append(token_exit_layers)

        for i in range(min(self.sandwich_full,batch_size)):
            token_exit_layers_batch[-1-i] = [self.num_hidden_layers-1] * len(token_exit_layers_batch[-1-i])


        token_exit_layers_wp_batch = []
        word_to_wordpiece_num_batch = batch_x['word_piece_num']

        for i,token_exit_layers in enumerate(token_exit_layers_batch):
            token_exit_layers_wp = inverse_pull_word_to_wordpiece(token_exit_layers,word_to_wordpiece_num_batch[i],num_hidden_layers=self.num_hidden_layers,
                                                                  copy_wordpiece=self.copy_wordpiece)
            token_exit_layers_wp_batch.append(token_exit_layers_wp)

        max_seq_len = batch_x['words'].size(1)
        max_wp_seq_len = batch_x['word_pieces'].size(1)
        seq_len = batch_x['seq_len']
        wp_seq_len = batch_x['word_piece_seq_len']
        should_exit_word_padded_tensor = torch.full([batch_size,12,max_seq_len],0)
        should_exit_wp_padded_tensor = torch.full([batch_size,12,max_wp_seq_len],0)
        # print('layers_mean:{}'.format(torch.mean(token_exit_layers_batch)))
        for i in range(batch_size):
            token_exit_layers = token_exit_layers_batch[i]
            token_exit_layers_wp = token_exit_layers_wp_batch[i]
            should_exit_word = exit_layers_to_should_exit(token_exit_layers,num_layers=self.num_hidden_layers)
            should_exit_wp = exit_layers_to_should_exit(token_exit_layers_wp,num_layers=self.num_hidden_layers)
            should_exit_word_padded_tensor[i,:,:seq_len[i]] = should_exit_word
            should_exit_wp_padded_tensor[i,:,:wp_seq_len[i]] = should_exit_wp

        # should_exit_word_padded_tensor = torch.ones_like(should_exit_word_padded_tensor)
        # should_exit_wp_padded_tensor = torch.ones_like(should_exit_wp_padded_tensor)


        should_exit_word_padded_tensor = should_exit_word_padded_tensor.to(seq_len)
        should_exit_wp_padded_tensor = should_exit_wp_padded_tensor.to(seq_len)



        batch_x['should_exit_word_padded_tensor'] = should_exit_word_padded_tensor.bool()
        batch_x['should_exit_wp_padded_tensor'] = should_exit_wp_padded_tensor

        return None


def cal_bert_flops(update_token_per_layer, num_hidden_layers=12,verbose=False):
    '''
    :param update_token_per_layer: for example , seq_len = 20 ,[20,18,14,10,5,2,0,0,0,0,0,0]，在第i层，还有多少token需要update
    :return:
    '''
    hidden_size = 768
    ffn_size = 3072

    def cal_self_attn_flops_layer(num_q,num_k_v):
        result = 0
        flops_w_k_v = num_k_v*(hidden_size**2)*2
        result+=flops_w_k_v

        flops_w_q = num_q*(hidden_size**2)
        result+=flops_w_q

        flops_self_attn = num_k_v*num_q*hidden_size*2
        result+=flops_self_attn

        flops_w_out = num_q*(hidden_size**2)

        result+=flops_w_out

        return result

    def cal_ffn_flops_layer(num_q):
        flops_ffn = num_q*3072*768*2

        return flops_ffn

    all_token_num = update_token_per_layer[0]
    all_flops = 0
    vanilla_bert_flops = 0
    for i in range(num_hidden_layers):
        if update_token_per_layer[i] == 0:
            break
        flops_self = cal_self_attn_flops_layer(update_token_per_layer[i],all_token_num)
        flops_ffn = cal_ffn_flops_layer(update_token_per_layer[i])

        all_flops+=(flops_ffn+flops_self)
        if i == 0:
            print(flops_ffn/1000000)
            print(flops_self/1000000)
            vanilla_bert_flops = (flops_ffn+flops_self)*12




    if verbose:
        print('{}:\nflops:{}\n vanilla flops:{}\nspeedup:{}'.format(update_token_per_layer,
                                                                    all_flops/(1000**2),
                                                                    vanilla_bert_flops/(1000**2),vanilla_bert_flops/all_flops))
    return all_flops

def filter_error_transitions(pred_last_layer,pred,constrain,constrain_inverse):
    '''

    :param pred_last_layer: seq_len,num_tags, after_softmax
    :param pred: the pred in 't_level_win_ee_copy'
    :return: seq_len,num_tags
    '''
    # # pred_label_last_layer = torch.argmax(pred_last_layer,dim=-1)
    # # pred_last_layer = nn.functional.softmax(pred_last_layer,dim=-1)
    # pred_uncertainty = get_uncertainty_2(pred_last_layer,need_softmax=False)
    # for i,p in enumerate(pred):
    #     if p is not None:
    #         pred_last_layer[i] = p
    # # should_consider = list(map(lambda x:x is None,pred))
    # # for i,sc in enumerate(should_consider):
    # #     if not sc:
    # #         pred_uncertainty[i] = 1000
    # _,the_most_certain_index = torch.min(pred_uncertainty, dim=0)
    # seq_len = pred_last_layer.size(0)
    #
    # for i in range(the_most_certain_index,seq_len):
    #     if pred[i] is not None:
    #         now_tag = torch.argmax(pred[i],dim=-1)
    #     else:
    #         now_tag = torch.argmax(pred_last_layer[i],dim=-1)
    #     if i+1<seq_len:
    #         pred_last_layer[i+1]+=constrain[now_tag]
    #
    # for i in reversed(range(0,the_most_certain_index)):
    #     if pred[i] is not None:
    #         now_tag = torch.argmax(pred[i],dim=-1)
    #     else:
    #         now_tag = torch.argmax(pred_last_layer[i],dim=-1)
    #     if i>0:
    #         pred_last_layer[i-1]+=constrain_inverse[now_tag]



    seq_len = pred_last_layer.size(0)
    for i in range(0,seq_len):
        if pred[i] is not None:
            now_tag = torch.argmax(pred[i],dim=-1)
        else:
            now_tag = torch.argmax(pred_last_layer[i],dim=-1)
        if i+1<seq_len:
            pred_last_layer[i+1]+=constrain[now_tag]

    # seq_len = pred_last_layer.size(0)
    # for i in range(0,seq_len-1):
    #     if pred[i] is not None:
    #         now_tag = torch.argmax(pred[i], dim=-1)
    #         pred_last_layer[i+1]+=constrain[now_tag]
    #     else:
    #         if pred[i+1] is not None:
    #             next_tag = torch.argmax(pred[i+1], dim=-1)
    #             pred_last_layer[i]+=constrain_inverse[next_tag]
    #         elif torch.max(pred_last_layer[i])>torch.max(pred_last_layer[i+1]):
    #             now_tag = torch.argmax(pred_last_layer[i], dim=-1)
    #             pred_last_layer[i + 1] += constrain[now_tag]
    #         else:
    #             next_tag = torch.argmax(pred_last_layer[i+1], dim=-1)
    #             pred_last_layer[i]+=constrain_inverse[next_tag]


    return pred_last_layer


class Twitter_Normalizer():

    def __init__(self):
        self.special_puncts = {"’": "'", "…": "..."}
        from emoji import demojize

        self.demojizer = demojize
    def normalizeToken(self, token):
        """
        Normalize tokens in a Tweet
        """
        lowercased_token = token.lower()
        if token.startswith("@") and len(token)>1:
            return "@user"
        elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
            return "http"
        elif len(token) == 1:
            if token in self.special_puncts:
                return self.special_puncts[token]
            if self.demojizer is not None:
                return self.demojizer(token)
            else:
                return token
        else:
            return token



if __name__ == '__main__':
    update_token_per_layer = [256]*3 + [128]*6 + [0]*3
    cal_bert_flops(update_token_per_layer,verbose=True)
    exit()
    from load_data import load_ontonotes4ner
    from paths import *
    x = ['O','B-per','I-per','B-org','B-loc','I-loc','I-loc']
    from fastNLP.io.pipe.utils import iob2bioes

    y = iob2bioes(x)
    print(y)
    exit()
    # bundle = load_ontonotes4ner(ontonote4ner_cn_path,index_token=True,char_min_freq=1,bigram_min_freq=1,norm_embed=True,
    #                             char_embedding_path=yangjie_rich_pretrain_unigram_path,bigram_embedding_path=yangjie_rich_pretrain_bigram_path,
    #                             # _cache_fp=cache_name,
    #                             _refresh=False)
    # new_bundle = transform_bmeso_bundle_to_bio(bundle)

    # tensor = torch.zeros(size=[2,7,4])
    # for i in range(tensor.size(0)):
    #     for j in range(tensor.size(1)):
    #         for k in range(tensor.size(2)):
    #             tensor[i,j,k] = (10*i+j)*10+k
    #
    # index = torch.tensor([[1,3],[6,4]])
    #
    # result = batch_index_select(tensor,index)
    # print(1)
    # print(result)
    pass




