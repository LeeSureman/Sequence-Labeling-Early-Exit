import os
import warnings
from itertools import chain
from functools import partial
import json
import numpy as np
import torch
from torch import nn

from fastNLP.embeddings.contextual_embedding import ContextualEmbedding
from fastNLP.core import logger
from fastNLP.core.vocabulary import Vocabulary
from fastNLP.io.file_utils import PRETRAINED_BERT_MODEL_DIR
from fastNLP.modules.encoder.bert import BertModel
from fastNLP.modules.tokenizer import BertTokenizer
from tmp_thop import myprofile


VOCAB_NAME = 'vocab.txt'
BERT_EMBED_HYPER = 'bert_hyper.json'
BERT_EMBED_FOLDER = 'bert'
BERT_ENCODER_HYPER = 'bert_hyper.json'
BERT_ENCODER_FOLDER = 'bert'

class _BertWordModel(nn.Module):
    def __init__(self, model_dir_or_name: str, vocab: Vocabulary, layers: str = '-1', pool_method: str = 'first',
                 include_cls_sep: bool = False, pooled_cls: bool = False, auto_truncate: bool = False, min_freq=2):
        super().__init__()

        if isinstance(layers, list):
            self.layers = [int(l) for l in layers]
        elif isinstance(layers, str):
            self.layers = list(map(int, layers.split(',')))
        else:
            raise TypeError("`layers` only supports str or list[int]")
        assert len(self.layers) > 0, "There is no layer selected!"

        neg_num_output_layer = -16384
        pos_num_output_layer = 0
        for layer in self.layers:
            if layer < 0:
                neg_num_output_layer = max(layer, neg_num_output_layer)
            else:
                pos_num_output_layer = max(layer, pos_num_output_layer)

        self.tokenzier = BertTokenizer.from_pretrained(model_dir_or_name)
        self.encoder = BertModel.from_pretrained(model_dir_or_name,
                                                 neg_num_output_layer=neg_num_output_layer,
                                                 pos_num_output_layer=pos_num_output_layer)
        self._max_position_embeddings = self.encoder.config.max_position_embeddings
        #  检查encoder_layer_number是否合理
        encoder_layer_number = len(self.encoder.encoder.layer)
        for layer in self.layers:
            if layer < 0:
                assert -layer <= encoder_layer_number, f"The layer index:{layer} is out of scope for " \
                                                       f"a bert model with {encoder_layer_number} layers."
            else:
                assert layer <= encoder_layer_number, f"The layer index:{layer} is out of scope for " \
                                                     f"a bert model with {encoder_layer_number} layers."

        assert pool_method in ('avg', 'max', 'first', 'last')
        self.pool_method = pool_method
        self.include_cls_sep = include_cls_sep
        self.pooled_cls = pooled_cls
        self.auto_truncate = auto_truncate

        # 将所有vocab中word的wordpiece计算出来, 需要额外考虑[CLS]和[SEP]
        self._has_sep_in_vocab = '[SEP]' in vocab  # 用来判断传入的数据是否需要生成token_ids

        word_to_wordpieces = []
        word_pieces_lengths = []
        for word, index in vocab:
            if index == vocab.padding_idx:  # pad是个特殊的符号
                word = '[PAD]'
            elif index == vocab.unknown_idx:
                word = '[UNK]'
            elif vocab.word_count[word]<min_freq:
                word = '[UNK]'
            word_pieces = self.tokenzier.wordpiece_tokenizer.tokenize(word)
            word_pieces = self.tokenzier.convert_tokens_to_ids(word_pieces)
            word_to_wordpieces.append(word_pieces)
            word_pieces_lengths.append(len(word_pieces))
        self._cls_index = self.tokenzier.vocab['[CLS]']
        self._sep_index = self.tokenzier.vocab['[SEP]']
        self._word_pad_index = vocab.padding_idx
        self._wordpiece_pad_index = self.tokenzier.vocab['[PAD]']  # 需要用于生成word_piece
        self.word_to_wordpieces = np.array(word_to_wordpieces)
        self.register_buffer('word_pieces_lengths', torch.LongTensor(word_pieces_lengths))
        logger.debug("Successfully generate word pieces.")

    def forward(self, words):
        r"""

        :param words: torch.LongTensor, batch_size x max_len
        :return: num_layers x batch_size x max_len x hidden_size或者num_layers x batch_size x (max_len+2) x hidden_size
        """
        with torch.no_grad():
            batch_size, max_word_len = words.size()
            word_mask = words.ne(self._word_pad_index)  # 为1的地方有word
            seq_len = word_mask.sum(dim=-1)
            batch_word_pieces_length = self.word_pieces_lengths[words].masked_fill(word_mask.eq(False),
                                                                                   0)  # batch_size x max_len
            word_pieces_lengths = batch_word_pieces_length.sum(dim=-1)  # batch_size
            max_word_piece_length = batch_word_pieces_length.sum(dim=-1).max().item()  # 表示word piece的长度(包括padding)
            if max_word_piece_length + 2 > self._max_position_embeddings:
                if self.auto_truncate:
                    word_pieces_lengths = word_pieces_lengths.masked_fill(
                        word_pieces_lengths + 2 > self._max_position_embeddings,
                        self._max_position_embeddings - 2)
                else:
                    raise RuntimeError(
                        "After split words into word pieces, the lengths of word pieces are longer than the "
                        f"maximum allowed sequence length:{self._max_position_embeddings} of bert. You can set "
                        f"`auto_truncate=True` for BertEmbedding to automatically truncate overlong input.")

            # +2是由于需要加入[CLS]与[SEP]
            word_pieces = words.new_full((batch_size, min(max_word_piece_length + 2, self._max_position_embeddings)),
                                         fill_value=self._wordpiece_pad_index)
            attn_masks = torch.zeros_like(word_pieces)
            # 1. 获取words的word_pieces的id，以及对应的span范围
            word_indexes = words.cpu().numpy()
            for i in range(batch_size):
                word_pieces_i = list(chain(*self.word_to_wordpieces[word_indexes[i, :seq_len[i]]]))
                if self.auto_truncate and len(word_pieces_i) > self._max_position_embeddings - 2:
                    word_pieces_i = word_pieces_i[:self._max_position_embeddings - 2]
                word_pieces[i, 1:word_pieces_lengths[i] + 1] = torch.LongTensor(word_pieces_i)
                attn_masks[i, :word_pieces_lengths[i] + 2].fill_(1)
            # 添加[cls]和[sep]
            word_pieces[:, 0].fill_(self._cls_index)
            batch_indexes = torch.arange(batch_size).to(words)
            word_pieces[batch_indexes, word_pieces_lengths + 1] = self._sep_index
            if self._has_sep_in_vocab:  # 但[SEP]在vocab中出现应该才会需要token_ids
                sep_mask = word_pieces.eq(self._sep_index).long()  # batch_size x max_len
                sep_mask_cumsum = sep_mask.flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
                token_type_ids = sep_mask_cumsum.fmod(2)
                token_type_ids = token_type_ids[:, :1].__xor__(token_type_ids)  # 如果开头是奇数，则需要flip一下结果，因为需要保证开头为0
            else:
                token_type_ids = torch.zeros_like(word_pieces)
        # 2. 获取hidden的结果，根据word_pieces进行对应的pool计算
        # all_outputs: [batch_size x max_len x hidden_size, batch_size x max_len x hidden_size, ...]
        print('word_pieces:{}'.format(word_pieces))
        bert_outputs, pooled_cls = self.encoder(word_pieces, token_type_ids=token_type_ids, attention_mask=attn_masks,
                                                output_all_encoded_layers=True)
        # print('bert_output:{} {}\n{}'.format(bert_outputs[0][-1].size(),bert_outputs[0][-1][0][:5][:5]))
        # print('bert_output:{} {}\n{}'.format(bert_outputs[0].size(),bert_outputs[1].size()))
        # print(bert_outputs)
        # print(bert_outputs)
        print('bert_outputs:')
        print(len(bert_outputs))
        print(bert_outputs[-1][0][:5][:5])
        # for i,lxn_tmp in bert_outputs:
        #     print('{}:{}'.format(i,lxn_tmp))

        # output_layers = [self.layers]  # len(self.layers) x batch_size x real_word_piece_length x hidden_size

        if self.include_cls_sep:
            s_shift = 1
            outputs = bert_outputs[-1].new_zeros(len(self.layers), batch_size, max_word_len + 2,
                                                     bert_outputs[-1].size(-1))

        else:
            s_shift = 0
            outputs = bert_outputs[-1].new_zeros(len(self.layers), batch_size, max_word_len,
                                                 bert_outputs[-1].size(-1))
        batch_word_pieces_cum_length = batch_word_pieces_length.new_zeros(batch_size, max_word_len + 1)
        batch_word_pieces_cum_length[:, 1:] = batch_word_pieces_length.cumsum(dim=-1)  # batch_size x max_len

        if self.pool_method == 'first':
            batch_word_pieces_cum_length = batch_word_pieces_cum_length[:, :seq_len.max()]
            batch_word_pieces_cum_length.masked_fill_(batch_word_pieces_cum_length.ge(max_word_piece_length), 0)
            _batch_indexes = batch_indexes[:, None].expand((batch_size, batch_word_pieces_cum_length.size(1)))
        elif self.pool_method == 'last':
            batch_word_pieces_cum_length = batch_word_pieces_cum_length[:, 1:seq_len.max()+1] - 1
            batch_word_pieces_cum_length.masked_fill_(batch_word_pieces_cum_length.ge(max_word_piece_length), 0)
            _batch_indexes = batch_indexes[:, None].expand((batch_size, batch_word_pieces_cum_length.size(1)))

        for l_index, l in enumerate(self.layers):
            output_layer = bert_outputs[l]
            real_word_piece_length = output_layer.size(1) - 2
            if max_word_piece_length > real_word_piece_length:  # 如果实际上是截取出来的
                paddings = output_layer.new_zeros(batch_size,
                                                  max_word_piece_length - real_word_piece_length,
                                                  output_layer.size(2))
                output_layer = torch.cat((output_layer, paddings), dim=1).contiguous()
            # 从word_piece collapse到word的表示
            truncate_output_layer = output_layer[:, 1:-1]  # 删除[CLS]与[SEP] batch_size x len x hidden_size
            if self.pool_method == 'first':
                tmp = truncate_output_layer[_batch_indexes, batch_word_pieces_cum_length]
                tmp = tmp.masked_fill(word_mask[:, :batch_word_pieces_cum_length.size(1), None].eq(False), 0)
                outputs[l_index, :, s_shift:batch_word_pieces_cum_length.size(1)+s_shift] = tmp

            elif self.pool_method == 'last':
                tmp = truncate_output_layer[_batch_indexes, batch_word_pieces_cum_length]
                tmp = tmp.masked_fill(word_mask[:, :batch_word_pieces_cum_length.size(1), None].eq(False), 0)
                outputs[l_index, :, s_shift:batch_word_pieces_cum_length.size(1)+s_shift] = tmp
            elif self.pool_method == 'max':
                for i in range(batch_size):
                    for j in range(seq_len[i]):
                        start, end = batch_word_pieces_cum_length[i, j], batch_word_pieces_cum_length[i, j + 1]
                        outputs[l_index, i, j + s_shift], _ = torch.max(truncate_output_layer[i, start:end], dim=-2)
            else:
                for i in range(batch_size):
                    for j in range(seq_len[i]):
                        start, end = batch_word_pieces_cum_length[i, j], batch_word_pieces_cum_length[i, j + 1]
                        outputs[l_index, i, j + s_shift] = torch.mean(truncate_output_layer[i, start:end], dim=-2)
            if self.include_cls_sep:
                if l in (len(bert_outputs) - 1, -1) and self.pooled_cls:
                    outputs[l_index, :, 0] = pooled_cls
                else:
                    outputs[l_index, :, 0] = output_layer[:, 0]
                outputs[l_index, batch_indexes, seq_len + s_shift] = output_layer[batch_indexes, word_pieces_lengths + s_shift]

        # 3. 最终的embedding结果
        return outputs

    def save(self, folder):
        """
        给定一个folder保存pytorch_model.bin, config.json, vocab.txt

        :param str folder:
        :return:
        """
        self.tokenzier.save_pretrained(folder)
        self.encoder.save_pretrained(folder)


class BertEmbedding(ContextualEmbedding):
    r"""
    使用BERT对words进行编码的Embedding。建议将输入的words长度限制在430以内，而不要使用512(根据预训练模型参数，可能有变化)。这是由于
    预训练的bert模型长度限制为512个token，而因为输入的word是未进行word piece分割的(word piece的分割有BertEmbedding在输入word
    时切分)，在分割之后长度可能会超过最大长度限制。

    BertEmbedding可以支持自动下载权重，当前支持的模型:
        en: base-cased
        en-base-uncased:
        en-large-cased-wwm:
        en-large-cased:
        en-large-uncased:
        en-large-uncased-wwm
        cn: 中文BERT wwm by HIT
        cn-base: 中文BERT base-chinese
        cn-wwm-ext: 中文BERT wwm by HIT with extra data pretrain.
        multi-base-cased: multilingual cased
        multi-base-uncased: multilingual uncased

    Example::

        >>> import torch
        >>> from fastNLP import Vocabulary
        >>> from fastNLP.embeddings import BertEmbedding
        >>> vocab = Vocabulary().add_word_lst("The whether is good .".split())
        >>> embed = BertEmbedding(vocab, model_dir_or_name='en-base-uncased', requires_grad=False, layers='4,-2,-1')
        >>> words = torch.LongTensor([[vocab.to_index(word) for word in "The whether is good .".split()]])
        >>> outputs = embed(words)
        >>> outputs.size()
        >>> # torch.Size([1, 5, 2304])
    """

    def __init__(self, vocab: Vocabulary, model_dir_or_name: str = 'en-base-uncased', layers: str = '-1',
                 pool_method: str = 'first', word_dropout=0, dropout=0, include_cls_sep: bool = False,
                 pooled_cls=True, requires_grad: bool = True, auto_truncate: bool = False, **kwargs):
        r"""

        :param ~fastNLP.Vocabulary vocab: 词表
        :param str model_dir_or_name: 模型所在目录或者模型的名称。当传入模型所在目录时，目录中应该包含一个词表文件(以.txt作为后缀名),
            权重文件(以.bin作为文件后缀名), 配置文件(以.json作为后缀名)。
        :param str layers: 输出embedding表示来自于哪些层，不同层的结果按照layers中的顺序在最后一维concat起来。以','隔开层数，层的序号是
            从0开始，可以以负数去索引倒数几层。 layer=0为embedding层（包括wordpiece embedding,
            position embedding和segment embedding）
        :param str pool_method: 因为在bert中，每个word会被表示为多个word pieces, 当获取一个word的表示的时候，怎样从它的word pieces
            中计算得到它对应的表示。支持 ``last`` , ``first`` , ``avg`` , ``max``。
        :param float word_dropout: 以多大的概率将一个词替换为unk。这样既可以训练unk也是一定的regularize。
        :param float dropout: 以多大的概率对embedding的表示进行Dropout。0.1即随机将10%的值置为0。
        :param bool include_cls_sep: bool，在bert计算句子的表示的时候，需要在前面加上[CLS]和[SEP], 是否在结果中保留这两个内容。 这样
            会使得word embedding的结果比输入的结果长两个token。如果该值为True，则在使用 :class::StackEmbedding 可能会与其它类型的
            embedding长度不匹配。
        :param bool pooled_cls: 返回的[CLS]是否使用预训练中的BertPool映射一下，仅在include_cls_sep时有效。如果下游任务只取[CLS]做预测，
            一般该值为True。
        :param bool requires_grad: 是否需要gradient以更新Bert的权重。
        :param bool auto_truncate: 当句子words拆分为word pieces长度超过bert最大允许长度(一般为512), 自动截掉拆分后的超过510个
            word pieces后的内容，并将第512个word piece置为[SEP]。超过长度的部分的encode结果直接全部置零。一般仅有只使用[CLS]
            来进行分类的任务将auto_truncate置为True。
        :param kwargs:
            int min_freq: 小于该次数的词会被unk代替, 默认为1
        """
        super(BertEmbedding, self).__init__(vocab, word_dropout=word_dropout, dropout=dropout)
        print('用了自己的bert embedding')
        if word_dropout > 0:
            assert vocab.unknown != None, "When word_drop>0, Vocabulary must contain the unknown token."

        if model_dir_or_name.lower() in PRETRAINED_BERT_MODEL_DIR:
            if 'cn' in model_dir_or_name.lower() and pool_method not in ('first', 'last'):
                logger.warning("For Chinese bert, pooled_method should choose from 'first', 'last' in order to achieve"
                               " faster speed.")
                warnings.warn("For Chinese bert, pooled_method should choose from 'first', 'last' in order to achieve"
                              " faster speed.")

        self._word_sep_index = -100
        if '[SEP]' in vocab:
            self._word_sep_index = vocab['[SEP]']
        self._word_cls_index = -100
        if '[CLS]' in vocab:
            self._word_cls_index = vocab['CLS']

        min_freq = kwargs.get('min_freq', 1)
        self._min_freq = min_freq
        self.model = _BertWordModel(model_dir_or_name=model_dir_or_name, vocab=vocab, layers=layers,
                                    pool_method=pool_method, include_cls_sep=include_cls_sep,
                                    pooled_cls=pooled_cls, min_freq=min_freq, auto_truncate=auto_truncate)

        self.requires_grad = requires_grad
        self._embed_size = len(self.model.layers) * self.model.encoder.hidden_size

    def _delete_model_weights(self):
        del self.model

    def forward(self, words):
        r"""
        计算words的bert embedding表示。计算之前会在每句话的开始增加[CLS]在结束增加[SEP], 并根据include_cls_sep判断要不要
            删除这两个token的表示。

        :param torch.LongTensor words: [batch_size, max_len]
        :return: torch.FloatTensor. batch_size x max_len x (768*len(self.layers))
        """
        words = self.drop_word(words)
        outputs = self._get_sent_reprs(words)
        if outputs is not None:
            return self.dropout(outputs)
        outputs = self.model(words)
        outputs = torch.cat([*outputs], dim=-1)

        return self.dropout(outputs)

    def drop_word(self, words):
        r"""
        按照设定随机将words设置为unknown_index。

        :param torch.LongTensor words: batch_size x max_len
        :return:
        """
        if self.word_dropout > 0 and self.training:
            with torch.no_grad():
                mask = torch.full_like(words, fill_value=self.word_dropout, dtype=torch.float, device=words.device)
                mask = torch.bernoulli(mask).eq(1)  # dropout_word越大，越多位置为1
                pad_mask = words.ne(self._word_pad_index)
                mask = pad_mask.__and__(mask)  # pad的位置不为unk
                if self._word_sep_index != -100:
                    not_sep_mask = words.ne(self._word_sep_index)
                    mask = mask.__and__(not_sep_mask)
                if self._word_cls_index != -100:
                    not_cls_mask = words.ne(self._word_cls_index)
                    mask = mask.__and__(not_cls_mask)
                words = words.masked_fill(mask, self._word_unk_index)
        return words

    def save(self, folder):
        """
        将embedding保存到folder这个目录下，将会保存三个文件vocab.txt, bert_embed_hyper.txt, bert_embed/, 其中bert_embed下包含
            config.json,pytorch_model.bin,vocab.txt三个文件(该folder下的数据也可以直接被BERTModel读取)

        :param str folder:
        :return:
        """
        os.makedirs(folder, exist_ok=True)

        self.get_word_vocab().save(os.path.join(folder, VOCAB_NAME))

        hyper = {}
        hyper['min_freq'] = self._min_freq
        hyper['layers'] = ','.join(map(str, self.model.layers))
        hyper['pool_method'] = self.model.pool_method
        hyper['dropout'] = self.dropout_layer.p
        hyper['word_dropout'] = self.word_dropout
        hyper['include_cls_sep'] = self.model.include_cls_sep
        hyper['pooled_cls'] = self.model.pooled_cls
        hyper['auto_truncate'] = self.model.auto_truncate
        hyper['requires_grad'] = bool(self.requires_grad)

        with open(os.path.join(folder, BERT_EMBED_HYPER), 'w', encoding='utf-8') as f:
            json.dump(hyper, f, indent=2)

        os.makedirs(os.path.join(folder, BERT_EMBED_FOLDER), exist_ok=True)
        self.model.save(os.path.join(folder, BERT_EMBED_FOLDER))
        logger.debug(f"BERTEmbedding has been saved in {folder}")

    @classmethod
    def load(cls, folder):
        """
        给定一个folder, 需要包含以下三个内容vocab.txt, bert_embed_hyper.txt, bert_embed/

        :param str folder:
        :return:
        """
        for name in [VOCAB_NAME, BERT_EMBED_FOLDER, BERT_EMBED_HYPER]:
            assert os.path.exists(os.path.join(folder, name)), f"{name} not found in {folder}."

        vocab = Vocabulary.load(os.path.join(folder, VOCAB_NAME))

        with open(os.path.join(folder, BERT_EMBED_HYPER), 'r', encoding='utf-8') as f:
            hyper = json.load(f)

        model_dir_or_name = os.path.join(os.path.join(folder, BERT_EMBED_FOLDER))

        bert_embed = cls(vocab=vocab, model_dir_or_name=model_dir_or_name, **hyper)
        return bert_embed

from fastNLP import SpanFPreRecMetric,AccuracyMetric

class MySpanFPreRecMetric(SpanFPreRecMetric):
    def __init__(self,layer_,*inp,**kwargs):
        is_main_two_stage = kwargs.get('is_main_two_stage',None)
        if 'is_main_two_stage' in kwargs:
            kwargs.pop('is_main_two_stage')
        super().__init__(*inp,**kwargs)
        self.layer_ = layer_
        self.is_main_two_stage = is_main_two_stage

    def evaluate(self,pred, target, seq_len):
        # print('self.layer_:{}'.format(self.layer_))
        if self.is_main_two_stage:
            if len(pred) == 1:
                if self.layer_ == 11:
                    pred = pred[-1]
                    super().evaluate(pred, target, seq_len)
                    return None
            else:
                pred = pred[self.layer_]
                super().evaluate(pred, target, seq_len)
                pass
            return None

        pred = pred[self.layer_]
        super().evaluate(pred,target,seq_len)


class MyAccuracyMetric(AccuracyMetric):
    def __init__(self,layer_,*inp,**kwargs):
        is_main_two_stage = kwargs.get('is_main_two_stage',None)
        if 'is_main_two_stage' in kwargs:
            kwargs.pop('is_main_two_stage')
        super().__init__(*inp,**kwargs)
        self.layer_ = layer_
        self.is_main_two_stage = is_main_two_stage

    def evaluate(self,pred, target, seq_len):
        if self.is_main_two_stage:
            if len(pred) == 1:
                if self.layer_ == 11:
                    pred = pred[-1]
                    super().evaluate(pred, target, seq_len)
                    return None
            else:
                pred = pred[self.layer_]
                super().evaluate(pred, target, seq_len)
                pass
            return None

        pred = pred[self.layer_]
        super().evaluate(pred,target,seq_len)

from fastNLP import WarmupCallback
import math
class MyWarmupCallback(WarmupCallback):

    def __init__(self, warmup=0.1, schedule='constant'):
        """

        :param int,float warmup: 如果warmup为int，则在该step之前，learning rate根据schedule的策略变化; 如果warmup为float，
            如0.1, 则前10%的step是按照schedule策略调整learning rate。
        :param str schedule: 以哪种方式调整。
            linear: 前warmup的step上升到指定的learning rate(从Trainer中的optimizer处获取的), 后warmup的step下降到0；
            constant前warmup的step上升到指定learning rate，后面的step保持learning rate.
        """
        super().__init__()
        self.warmup = max(warmup, 0.)

        self.initial_lrs = []  # 存放param_group的learning rate
        if schedule == 'constant':
            self.get_lr = self._get_constant_lr
        elif schedule == 'linear':
            self.get_lr = self._get_linear_lr
        elif schedule == 'inverse_square':
            self.get_lr = self._get_inverse_square_lr
        else:
            raise RuntimeError("Only support 'linear', 'constant'.")

    def _get_inverse_square_lr(self, progress):
        if progress<self.warmup:
            return progress/self.warmup
        return max((math.sqrt(progress) - 1.) / (math.sqrt(self.warmup) - 1.), 0.)


from fastNLP.core.batch import DataSetIter,BatchIter,DataSetGetter,_to_tensor
from fastNLP import DataSet
from collections import defaultdict

def _pad(batch_dict, dataset, as_numpy):
    result = {}
    for n, vlist in batch_dict.items():
        f = dataset.field_arrays[n]
        if f.padder is None:
            result[n] = vlist
        else:
            res = f.pad(vlist)
            if not as_numpy:
                res, _ = _to_tensor(res, field_dtype=f.dtype)
            result[n] = res

    return result

class MyDatasetGetter(DataSetGetter):
    def collate_fn(self, ins_list: list):
        r"""

        :param batch: [[idx1, x_dict1, y_dict1], [idx2, x_dict2, y_dict2], [xx, xx, xx]]
        :return:
        """
        indices = []
        sin_x, sin_y = defaultdict(list), defaultdict(list)
        # 收集需要关注的field的数据
        for idx, ins in ins_list:
            indices.append(idx)
            for n, v in ins.items():
                if n in self.x_names:
                    sin_x[n].append(v)
                if n in self.y_names:
                    sin_y[n].append(v)
        # 根据情况，进行pad
        sin_x = _pad(sin_x, dataset=self.dataset, as_numpy=self.as_numpy)
        sin_y = _pad(sin_y, dataset=self.dataset, as_numpy=self.as_numpy)

        if not self.dataset.collater.is_empty():
            bx, by = self.dataset._collate_batch(ins_list)
            sin_x.update(bx)
            sin_y.update(by)

        return indices, sin_x, sin_y


class MyDatasetIter(DataSetIter):
    def __init__(self, dataset, batch_size=1, sampler=None, as_numpy=False, num_workers=0, pin_memory=False,
                 drop_last=False, timeout=0, worker_init_fn=None, batch_sampler=None):
        r"""

        :param dataset: :class:`~fastNLP.DataSet` 对象, 数据集
        :param int batch_size: 取出的batch大小
        :param sampler: 规定使用的 :class:`~fastNLP.Sampler` 方式. 若为 ``None`` , 使用 :class:`~fastNLP.SequentialSampler`.

            Default: ``None``
        :param bool as_numpy: 若为 ``True`` , 输出batch为 numpy.array. 否则为 :class:`torch.Tensor`.

            Default: ``False``
        :param int num_workers: 使用多少个进程来预处理数据
        :param bool pin_memory: 是否将产生的tensor使用pin memory, 可能会加快速度。
        :param bool drop_last: 如果最后一个batch没有batch_size这么多sample，就扔掉最后一个
        :param timeout: 生成一个batch的timeout值
        :param worker_init_fn: 在每个worker启动时调用该函数，会传入一个值，该值是worker的index。
        :param batch_sampler: 当每次batch取出的数据数量不一致时，可以使用该sampler。batch_sampler每次iter应该输出一个list的index。
            当batch_sampler不为None时，参数batch_size, sampler, drop_last会被忽略。
        """
        assert isinstance(dataset, DataSet)
        dataset = MyDatasetGetter(dataset, as_numpy)
        collate_fn = dataset.collate_fn
        if batch_sampler is not None:
            batch_size = 1
            sampler = None
            drop_last = False
        BatchIter.__init__(self,
            dataset=dataset, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=pin_memory,
            drop_last=drop_last, timeout=timeout, worker_init_fn=worker_init_fn,
            collate_fn=collate_fn, batch_sampler=batch_sampler
        )


from fastNLP import Tester
from fastNLP.core.utils import _move_dict_value_to_device,_get_model_device,_get_func_signature,_CheckError,_check_loss_evaluate,_build_args
from tqdm import tqdm
import time
# from thop import profile
from tmp_thop import myprofile
class MyTester(Tester):
    def __init__(self,*input,**kwargs):
        super().__init__(*input,**kwargs)
        if kwargs.get('record_flops') is None:
            self.record_flops = False
        else:
            self.record_flops = kwargs['record_flops']

        if kwargs.get('custom_ops') is None:
            self.custom_ops = dict()
        else:
            self.custom_ops = kwargs['custom_ops']
        self.total_flops = 0

        self.preds = []


    # def test(self):
    #     r"""开始进行验证，并返回验证结果。
    #
    #     :return Dict[Dict]: dict的二层嵌套结构，dict的第一层是metric的名称; 第二层是这个metric的指标。一个AccuracyMetric的例子为{'AccuracyMetric': {'acc': 1.0}}。
    #     """
    #     # turn on the testing mode; clean up the history
    #     self._model_device = _get_model_device(self._model)
    #     network = self._model
    #     self._mode(network, is_test=True)
    #     data_iterator = self.data_iterator
    #     eval_results = {}
    #     try:
    #         with torch.no_grad():
    #             if not self.use_tqdm:
    #                 from .utils import _pseudo_tqdm as inner_tqdm
    #             else:
    #                 inner_tqdm = tqdm
    #             with inner_tqdm(total=len(data_iterator), leave=False, dynamic_ncols=True) as pbar:
    #                 pbar.set_description_str(desc="Test")
    #
    #                 start_time = time.time()
    #
    #                 for batch_x, batch_y in data_iterator:
    #                     _move_dict_value_to_device(batch_x, batch_y, device=self._model_device)
    #                     pred_dict = self._data_forward(self._predict_func, batch_x)
    #                     if not isinstance(pred_dict, dict):
    #                         raise TypeError(f"The return value of {_get_func_signature(self._predict_func)} "
    #                                         f"must be `dict`, got {type(pred_dict)}.")
    #                     for metric in self.metrics:
    #                         metric(pred_dict, batch_y)
    #
    #                     if self.use_tqdm:
    #                         pbar.update()
    #
    #                 for metric in self.metrics:
    #                     eval_result = metric.get_metric()
    #                     if not isinstance(eval_result, dict):
    #                         raise TypeError(f"The return value of {_get_func_signature(metric.get_metric)} must be "
    #                                         f"`dict`, got {type(eval_result)}")
    #                     metric_name = metric.get_metric_name()
    #                     eval_results[metric_name] = eval_result
    #                 pbar.close()
    #                 end_time = time.time()
    #                 test_str = f'Evaluate data in {round(end_time - start_time, 2)} seconds!'
    #                 if self.verbose >= 0:
    #                     self.logger.info(test_str)
    #     except _CheckError as e:
    #         prev_func_signature = _get_func_signature(self._predict_func)
    #         _check_loss_evaluate(prev_func_signature=prev_func_signature, func_signature=e.func_signature,
    #                              check_res=e.check_res, pred_dict=pred_dict, target_dict=batch_y,
    #                              dataset=self.data, check_level=0)
    #     finally:
    #         self._mode(network, is_test=False)
    #     if self.verbose >= 1:
    #         logger.info("[tester] \n{}".format(self._format_eval_results(eval_results)))
    #     return eval_results

    def _data_forward(self, func, x):
        r"""A forward pass of the model. """
        x = _build_args(func, **x)
        y = self._predict_func_wrapper(**x)
        self.preds.append(y)
        if self.record_flops:
            flops,params = myprofile(model=self._model,inputs=x,verbose=False,custom_ops=self.custom_ops)
            self.total_flops+=flops
        return y



from fastNLP.io import ConllLoader
from fastNLP import Instance


def _my_read_conll(path, encoding='utf-8',sep=None, indexes=None, dropna=True,comment_tag=None):
    r"""
    Construct a generator to read conll items.

    :param path: file path
    :param encoding: file's encoding, default: utf-8
    :param sep: seperator
    :param indexes: conll object's column indexes that needed, if None, all columns are needed. default: None
    :param dropna: weather to ignore and drop invalid data,
            :if False, raise ValueError when reading invalid data. default: True
    :return: generator, every time yield (line number, conll item)
    """

    def parse_conll(sample):
        sample = list(map(list, zip(*sample)))
        sample = [sample[i] for i in indexes]
        for f in sample:
            if len(f) <= 0:
                raise ValueError('empty field')
        return sample

    with open(path, 'r', encoding=encoding) as f:
        sample = []
        start = next(f).strip()
        if start != '':
            sample.append(start.split(sep)) if sep else sample.append(start.split())
        for line_idx, line in enumerate(f, 1):
            line = line.strip()
            if line == '':
                if len(sample):
                    try:
                        res = parse_conll(sample)
                        sample = []
                        yield line_idx, res
                    except Exception as e:
                        if dropna:
                            logger.warning('Invalid instance which ends at line: {} has been dropped.'.format(line_idx))
                            sample = []
                            continue
                        raise ValueError('Invalid instance which ends at line: {}'.format(line_idx))
            elif comment_tag is not None and line.startswith(comment_tag):
                continue
            else:
                sample.append(line.split(sep)) if sep else sample.append(line.split())
        if len(sample) > 0:
            try:
                res = parse_conll(sample)
                yield line_idx, res
            except Exception as e:
                if dropna:
                    return
                logger.error('invalid instance ends at line: {}'.format(line_idx))
                raise e

class MyConllLoader(ConllLoader):
    def _load(self, path):
        r"""
        传入的一个文件路径，将该文件读入DataSet中，field由ConllLoader初始化时指定的headers决定。

        :param str path: 文件的路径
        :return: DataSet
        """
        ds = DataSet()
        for idx, data in _my_read_conll(path,sep=self.sep, indexes=self.indexes, dropna=self.dropna):
            ins = {h: data[i] for i, h in enumerate(self.headers)}
            ds.append(Instance(**ins))
        return ds



# from fastNLP import FitlogCallback
# class MyFitlogCallback(FitlogCallback):
#     def