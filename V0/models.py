import torch
import torch.nn as nn
from utils import get_ptm_from_name,get_crf_zero_init
from fastNLP import seq_len_to_mask
from fastNLP.modules import ConditionalRandomField
from utils import MyDropout
from transformers import AutoModel,AutoConfig,AutoTokenizer,AlbertTokenizerFast,AlbertTokenizer,AlbertModel,AlbertConfig,BertConfig,BertModel,BertTokenizer
# from transformers.modeling_bert import BertEncoder
from modules import TransformerEncoder
from utils import batch_index_select_yf
from fastNLP.embeddings import CNNCharEmbedding


# class TENER_seq_label()
class BertEncoder_EE(BertEncoder):
    def adaptive_forward(self, hidden_states, current_layer, attention_mask=None, head_mask=None):
        layer_outputs = self.layer[current_layer](hidden_states, attention_mask, head_mask[current_layer])

        hidden_states = layer_outputs[0]

        return hidden_states



class BertModel_EE(BertModel):
    """
    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`; an
    :obj:`encoder_hidden_states` is expected as an input to the forward pass.
    .. _`Attention is all you need`:
        https://arxiv.org/abs/1706.03762
    """

    def __init__(self, config):
        super().__init__(config)

        self.encoder = BertEncoder_EE(config)

        self.init_weights()
        # self.patience = 0
        self.inference_instances_num = 0
        self.inference_layers_num = 0

        # self.regression_threshold = 0

    # def set_regression_threshold(self, threshold):
    #     self.regression_threshold = threshold

    # def set_patience(self, patience):
    #     self.patience = patience

    def reset_stats(self):
        self.inference_instances_num = 0
        self.inference_layers_num = 0

    def log_stats(self):
        avg_inf_layers = self.inference_layers_num / self.inference_instances_num
        message = f"*** Patience = {self.patience} Avg. Inference Layers = {avg_inf_layers:.2f} Speed Up = {1 - avg_inf_layers / self.config.num_hidden_layers:.2f} ***"
        print(message)

    # @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_dropout=None,
        output_layers=None,
        regression=False,
    ):
        r"""
        Return:
            :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
            last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
                Sequence of hidden-states at the output of the last layer of the model.
            pooler_output (:obj:`torch.FloatTensor`: of shape :obj:`(batch_size, hidden_size)`):
                Last layer hidden-state of the first token of the sequence (classification token)
                further processed by a Linear layer and a Tanh activation function. The Linear
                layer weights are trained from the next sentence prediction (classification)
                objective during pre-training.
                This output is usually *not* a good summary
                of the semantic content of the input, you're often better with averaging or pooling
                the sequence of hidden-states for the whole input sequence.
            hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
                Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
                of shape :obj:`(batch_size, sequence_length, hidden_size)`.
                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
                Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
                :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
                heads.
        """

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = embedding_output

        if self.training:
            res = []
            for i in range(self.config.num_hidden_layers):
                encoder_outputs = self.encoder.adaptive_forward(
                    encoder_outputs, current_layer=i, attention_mask=extended_attention_mask, head_mask=head_mask
                )

                pooled_output = self.pooler(encoder_outputs)
                logits = output_layers[i](output_dropout(pooled_output))
                res.append(logits)
        elif self.patience == 0:  # Use all layers for inference
            encoder_outputs = self.encoder(
                embedding_output,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
            )
            pooled_output = self.pooler(encoder_outputs[0])
            res = [output_layers[self.config.num_hidden_layers - 1](pooled_output)]
        else:
            patient_counter = 0
            patient_result = None
            calculated_layer_num = 0
            for i in range(self.config.num_hidden_layers):
                calculated_layer_num += 1
                encoder_outputs = self.encoder.adaptive_forward(
                    encoder_outputs, current_layer=i, attention_mask=extended_attention_mask, head_mask=head_mask
                )

                pooled_output = self.pooler(encoder_outputs)
                logits = output_layers[i](pooled_output)
                if regression:
                    labels = logits.detach()
                    if patient_result is not None:
                        patient_labels = patient_result.detach()
                    if (patient_result is not None) and torch.abs(patient_result - labels) < self.regression_threshold:
                        patient_counter += 1
                    else:
                        patient_counter = 0
                else:
                    labels = logits.detach().argmax(dim=1)
                    if patient_result is not None:
                        patient_labels = patient_result.detach().argmax(dim=1)
                    if (patient_result is not None) and torch.all(labels.eq(patient_labels)):
                        patient_counter += 1
                    else:
                        patient_counter = 0

                patient_result = logits
                if patient_counter == self.patience:
                    break
            res = [patient_result]
            self.inference_layers_num += calculated_layer_num
            self.inference_instances_num += 1

        return res

class BERT_SeqLabel_EE(nn.Module):
    def __init__(self,bundle,args,):
        super().__init__()
        if hasattr(bundle,'embeddings'):
            self.bigram_embedding = bundle.embeddings['bigram']
        else:
            self.bigram_embedding = None
        if args.use_char:
            self.cnn_char = CNNCharEmbedding(bundle.vocabs['words'],word_dropout=0.01)
        self.vocabs = bundle.vocabs
        self.use_crf = args.use_crf
        self.bundle = bundle
        self.args = args
        # self.ptm_encoder_name = args.ptm_encoder_name
        self.loss_func = nn.CrossEntropyLoss(ignore_index=-100)
        # self.ptm_encoder = get_ptm_from_name(args.ptm_name,self.vocabs['words'],args.ptm_pool_method,
        #                                      args.ptm_word_dropout,layers=args.ptm_layers)
        if args.use_fastnlp_bert:
            print('use fastnlp bert!')
            self.ptm_encoder = get_ptm_from_name('bert_cn-wwm', self.vocabs['words'], args.ptm_pool_method,
                                                 0.01, layers=[-1])
        else:
            if 'albert' in args.ptm_name:
                self.ptm_config = AlbertConfig.from_pretrained(args.ptm_name)
                self.ptm_encoder = AlbertModel.from_pretrained(args.ptm_name)
            elif 'roberta' in args.ptm_name:
                raise NotImplementedError
            elif 'bert' in args.ptm_name:
                self.ptm_config = BertConfig.from_pretrained(args.ptm_name)
                self.ptm_encoder = BertModel.from_pretrained(args.ptm_name)
                # raise NotImplementedError
            else:
                raise NotImplementedError
            # print('ptm_model:{}'.format(self.ptm_encoder))
            self.ptm_encoder.resize_token_embeddings(new_num_tokens=len(bundle.tokenizer))
        if args.use_pytorch_dropout:
            Dropout_CLASS = nn.Dropout
        else:
            Dropout_CLASS = MyDropout
        # self.ptm_dropout =
        # self.ptm_dropout = Dropout_CLASS(args.ptm_dropout)
        target_size = len(self.vocabs['target'])
        if self.vocabs['target'].padding_idx:
            target_size-=1
        if self.vocabs['target'].unknown_idx:
            target_size-=1

        # self.w_out = nn.Linear(self.ptm_encoder._embed_size,target_size)
        # self.classifier = TENER()

        if args.after_bert == 'tener':
            w_dim_inp_size = self.ptm_config.hidden_size
            if self.args.use_bigram and self.bigram_embedding:
                w_dim_inp_size = self.ptm_config.hidden_size + self.bigram_embedding._embed_size
            if self.args.use_char:
                w_dim_inp_size = self.ptm_config.hidden_size + self.cnn_char._embed_size


            self.w_dim = nn.Linear(w_dim_inp_size, args.cls_hidden)
            self.transformer_cls = TransformerEncoder(1,args.cls_hidden,args.cls_head,args.cls_ff*args.cls_hidden,dropout=args.cls_dropout,
                                                 after_norm=args.cls_after_norm,attn_type='adatrans',scale=args.cls_scale,dropout_attn=args.cls_drop_attn,
                                                 pos_embed=None,target_size=target_size)
            # self.cls_out = nn.Sequential([
            #     nn.Linear(self.ptm_config.hidden_size,args.cls_hidden),
            #     transformer_cls,
            # ])
            self.cls_out = lambda x,mask: self.transformer_cls(self.w_dim(x),mask)
        elif args.after_bert == 'linear':
            if args.use_fastnlp_bert:
                self.cls_out = nn.Linear(768, target_size)
            else:
                self.cls_out = nn.Linear(self.ptm_config.hidden_size,target_size)

        if self.use_crf:
            self.crf = get_crf_zero_init(target_size,True)


        self.embed_dropout = nn.Dropout(args.embed_dropout)



    def forward(self,**kwargs):
        # print(kwargs.keys())
        words = kwargs['words']
        seq_len = kwargs['seq_len']
        target = kwargs['target']

        word_pieces = kwargs['word_pieces']
        attention_mask = kwargs['bert_attention_mask']

        # print('word_pieces:{}'.format(word_pieces.size()))

        ptm_pool_pos = None
        if self.args.ptm_pool_method == 'first':
            ptm_pool_pos = kwargs['first_word_pieces_pos']
        elif self.args.ptm_pool_method == 'first_skip_space':
            ptm_pool_pos = kwargs['first_word_pieces_pos_skip_space']
        elif self.args.ptm_pool_method == 'last':
            ptm_pool_pos = kwargs['last_word_pieces_pos']


        if self.args.use_fastnlp_bert:
            ptm_output = self.ptm_encoder(words)
            encoded = ptm_output
        else:
            ptm_output = self.ptm_encoder(input_ids=word_pieces,attention_mask=attention_mask,token_type_ids=None,position_ids=None)
        # print('encoded:{}'.format(type(encoded)))
        # print('encoded:{}'.format(encoded))
        # print('last_hidden_state:{}'.format(encoded.last_hidden_state.size()))
        # print('pooler_output:{}'.format(encoded.pooler_output.size()))
        # ptm_output = encoded
            encoded = ptm_output.last_hidden_state
            encoded = batch_index_select_yf(encoded, ptm_pool_pos)

        # print('word_pieces:{}'.format(word_pieces))
        # print('encoded:{}\n{}'.format(encoded.size(),encoded[0][:5][:5]))
        # exit()

        # if self.bigram_embedding:

        if self.args.after_bert == 'tener':
            # print('encoded:{}'.format(encoded.size()))
            # print('attn_mask:{}'.format(attention_mask.size()))
            if self.args.use_bigram and self.bigram_embedding:
                bigrams = kwargs['bigrams']
                bigrams = self.bigram_embedding(bigrams)
                encoded = torch.cat([encoded,bigrams],dim=-1)
            if self.args.use_char:
                char_encoded = self.cnn_char(words)
                encoded = torch.cat([encoded,char_encoded],dim=-1)
            encoded = self.embed_dropout(encoded)
            tener_attn_mask = seq_len_to_mask(seq_len)
            pred = self.cls_out(encoded,tener_attn_mask)
        elif self.args.after_bert == 'linear':
            pred = self.cls_out(encoded)

        # encoded = batch_index_select_yf(encoded, ptm_pool_pos)
        # print('pred:{}'.format(pred.size()))
        # print('ptm_pool_pos:{}'.format(ptm_pool_pos))
        # pred = batch_index_select_yf(pred,ptm_pool_pos)


        result = {}

        if self.use_crf:
            mask = seq_len_to_mask(seq_len)
            if self.training:
                loss = self.crf(pred,target,mask).mean(dim=0)
                result['loss'] = loss
            else:
                pred, path = self.crf.viterbi_decode(pred,mask)
                result['pred'] = pred

        else:

            if self.training:
                pred = pred.flatten(0,1)
                target = target.flatten(0,1)
                loss = self.loss_func(pred,target)
                result['loss'] = loss
            else:
                result['pred'] = pred
        return result





