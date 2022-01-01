import torch
import torch.nn as nn
from utils import get_ptm_from_name,get_crf_zero_init
from fastNLP import seq_len_to_mask
from fastNLP.modules import ConditionalRandomField
from utils import MyDropout
from transformers import AutoModel,AutoConfig,AutoTokenizer,AlbertTokenizerFast,AlbertTokenizer,AlbertModel,AlbertConfig,BertConfig,BertModel,BertTokenizer
from transformers.models.bert.modeling_bert import BertLayer,BertEncoder,BertModel,BertAttention,BertSelfAttention,BertOutput

from modules import TransformerEncoder
from utils import batch_index_select_yf
from fastNLP.embeddings import CNNCharEmbedding
from transformers.modeling_utils import apply_chunking_to_forward
from utils import get_uncertainty,get_entropy
from utils import mask_logit_by_certain_pred_and_constrain
from utils import filter_error_transitions

class Pseudo_Classifier(nn.Module):
    def __init__(self, args, input_size, target_size):
        super().__init__()
        # w_dim_inp_size = input_size
        self.input_size = input_size
        self.dropout = nn.Dropout(args.embed_dropout)
        self.args = args
        self.target_size = target_size

        if args.after_bert == 'linear':
            self.w = nn.Linear(input_size,target_size)
        elif args.after_bert == 'non_linear':
            self.w1 = nn.Linear(input_size,input_size)
            self.w2 = nn.Linear(input_size,target_size)
        else:
            raise NotImplementedError

    def forward(self, hidden, mask,label_feature=None):
        if self.args.after_bert == 'linear':
            hidden = self.dropout(hidden)
            hidden = self.w(hidden)
            return hidden
        elif self.args.after_bert == 'non_linear':
            raise NotImplementedError
            hidden_original = hidden
            hidden = self.w1(hidden)
            hidden = self.dropout(hidden)
            hidden = hidden + hidden_original
            hidden = self.w2(hidden)
            return hidden

class TENER_classifier(nn.Module):
    def __init__(self, args, input_size, target_size):
        super().__init__()
        # w_dim_inp_size = input_size
        self.input_size = input_size
        self.embed_dropout = nn.Dropout(args.embed_dropout)
        self.w_dim = nn.Linear(input_size, args.cls_hidden)
        self.transformer_cls = TransformerEncoder(1, args.cls_hidden, args.cls_head, args.cls_ff * args.cls_hidden,
                                                  dropout=args.cls_dropout,
                                                  after_norm=args.cls_after_norm, attn_type='adatrans',
                                                  scale=args.cls_scale, dropout_attn=args.cls_drop_attn,
                                                  pos_embed=None, target_size=target_size)
        self.target_size = target_size

        self.init_parameters()


    def forward(self, hidden, mask,label_feature=None):
        hidden = self.embed_dropout(hidden)
        hidden = self.w_dim(hidden)

        if label_feature is not None:
            hidden+=label_feature

        pred = self.transformer_cls(hidden,mask)

        return pred

    def init_parameters(self):
        with torch.no_grad():
            print('{}init pram{}'.format('*' * 15, '*' * 15))
            for n, p in self.named_parameters():
                if 'embedding' not in n and 'pos' not in n and 'pe' not in n \
                        and 'bias' not in n and 'crf' not in n and p.dim() > 1:
                    try:
                        nn.init.xavier_uniform_(p)
                        print('xavier uniform init:{}'.format(n))
                        # if args.init == 'uniform':
                        #     nn.init.xavier_uniform_(p)
                        #     print_info('xavier uniform init:{}'.format(n))
                        # elif args.init == 'norm':
                        #     print_info('xavier norm init:{}'.format(n))
                        #     nn.init.xavier_normal_(p)
                    except Exception as e:
                        print(e)
                        print(n)
                        exit(1208)
            print('{}init pram{}'.format('*' * 15, '*' * 15))

class BertOutput_EE(BertOutput):
    def __init__(self,config):
        super().__init__(config)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        # hidden_states = self.LayerNorm(hidden_states)
        # hidden_states = self.LayerNorm(hidden_states)
        # hidden_states = self.LayerNorm(hidden_states)
        return hidden_states



class BertLayer_EE(BertLayer):
    def __init__(self, config,args):
        super().__init__(config)
        self.args = args
        self.output = BertOutput_EE(config)
        # self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # self.seq_len_dim = 1
        # self.attention = BertAttention_EE(config)
        # self.is_decoder = config.is_decoder
        # self.add_cross_attention = config.add_cross_attention
        # if self.add_cross_attention:
        #     raise NotImplementedError
        #     assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
        #     self.crossattention = BertAttention(config)
        # self.intermediate = BertIntermediate(config)
        # self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        # last_layer_norm=None,
        should_update_indices=None
    ):
        # if self.args.keep_norm_same and last_layer_norm is not None:
        #     # print('keep norm same!')
        #     hidden_states = last_layer_norm(hidden_states)
        q_hidden_states = hidden_states
        k_v_hidden_states = hidden_states

        if should_update_indices is not None:
            assert q_hidden_states.size(0) == 1
            q_hidden_states = q_hidden_states[:,should_update_indices]

        if False:
            raise NotImplementedError
            q_hidden_states = -1

        # self_attention_outputs = self.attention(
        #     hidden_states=q_hidden_states,
        #     # attention_mask,
        #     head_mask=head_mask,
        #     output_attentions=output_attentions,
        #     encoder_hidden_states=k_v_hidden_states,
        #     encoder_attention_mask=attention_mask
        # )
        self_attention_outputs = self.attention(
            q_hidden_states,
            # attention_mask,
            None,
            head_mask,
            k_v_hidden_states,
            attention_mask,
            output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            raise NotImplementedError
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        # if not self.args.keep_norm_same:
        #     # print('not keep norm same!')
        #     layer_output = self.output.LayerNorm(layer_output)

        if False:
            #把更新的和copy的hidden_state放到一起
            pass

        # if self.args.keep_norm_same:
        #     layer_output = self.output.LayerNorm(layer_output)

        outputs = (layer_output,) + outputs

        return outputs

# class TENER_seq_label()
class BertEncoder_EE(BertEncoder):
    def __init__(self, config,args):
        super().__init__(config)
        # self.output_attentions = config.output_attentions
        # self.output_hidden_states = config.output_hidden_states
        self.args = args
        self.layer = nn.ModuleList([BertLayer_EE(config,args) for _ in range(config.num_hidden_layers)])
        # self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])


    def adaptive_forward(self, hidden_states, current_layer, attention_mask=None, head_mask=None, should_update_indices=None):
        # if self.args.keep_norm_same and current_layer!=0:
        #     last_layer_norm = self.layer[current_layer-1].output.LayerNorm
        # else:
        #     last_layer_norm = None
        layer_outputs = self.layer[current_layer](hidden_states, attention_mask, head_mask[current_layer],should_update_indices=should_update_indices)

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

    def __init__(self, config,args):
        super().__init__(config)
        self.args = args
        self.encoder = BertEncoder_EE(config,args)

        self.init_weights()
        # self.patience = 0
        self.inference_instances_num = 0
        self.inference_layers_num = 0
        self.exit_layer_num = [0] * 12
        self.exit_layer_num_token = [0]*12
        self.inference_token_num = 0

        # self.regression_threshold = 0

    # def set_regression_threshold(self, threshold):
    #     self.regression_threshold = threshold

    # def set_patience(self, patience):
    #     self.patience = patience

    def reset_stats(self):

        self.inference_instances_num = 0
        self.inference_layers_num = 0
        self.exit_layer_num = [0] * 12
        self.exit_layer_num_token = [0]*12
        self.inference_token_num = 0

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
        ptm_pool_pos=None,
        extra_token_feature=None,
        word_mask=None,
        should_exit_word_padded_tensor=None,
        should_exit_wp_padded_tensor=None,
        **kwargs
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
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids
        )
        encoder_outputs = embedding_output

        if self.training:
            if self.args.train_mode == 'joint_sample_copy':
                should_exit_wp_padded_tensor = should_exit_wp_padded_tensor.unsqueeze(-1)
                should_exit_word_padded_tensor = should_exit_word_padded_tensor.unsqueeze(-1)

                last_hidden_state = embedding_output
                last_hidden_state_before_layernorm = embedding_output

                res = []

                # if 'copy' in self.args.train_mode:
                #     last_layer_hidden_state = embedding_output

                    # raise NotImplementedError
                for i in range(self.config.num_hidden_layers):
                    encoder_outputs = self.encoder.adaptive_forward(
                        last_hidden_state, current_layer=i, attention_mask=extended_attention_mask, head_mask=head_mask
                    )
                    if not self.args.keep_norm_same:
                        # print('not keep norm same')
                        encoder_outputs = self.encoder.layer[i].output.LayerNorm(encoder_outputs)
                        last_hidden_state = \
                            (should_exit_wp_padded_tensor[:,i] * encoder_outputs) + \
                            (1 - should_exit_wp_padded_tensor[:,i]) * last_hidden_state
                        encoder_outputs = last_hidden_state
                    else:
                        last_hidden_state_before_layernorm = \
                            (should_exit_wp_padded_tensor[:,i] * encoder_outputs) + \
                            (1 - should_exit_wp_padded_tensor[:,i]) * last_hidden_state_before_layernorm
                        last_hidden_state = self.encoder.layer[i].output.LayerNorm(last_hidden_state_before_layernorm)
                        encoder_outputs = last_hidden_state

                    # if self.args.keep_norm_same:
                    #     # print('keep norm same')
                    #     last_hidden_state = self.encoder.layer[i].output.LayerNorm(last_hidden_state_maybe_before_layernorm)
                    # else:
                    #     last_hidden_state = last_hidden_state_maybe_before_layernorm
                    #
                    # encoder_outputs = last_hidden_state

                    assert ptm_pool_pos is not None

                    assert word_mask is not None


                    if 'joint' in self.args.train_mode:
                        encoder_outputs_for_cls = batch_index_select_yf(encoder_outputs, ptm_pool_pos)
                        if extra_token_feature is not None:
                            encoder_outputs_for_cls = torch.cat([encoder_outputs_for_cls, extra_token_feature], dim=-1)
                        logits = output_layers[i](encoder_outputs_for_cls,word_mask)
                        res.append(logits)







                # if not self.args.keep_norm_same:
                #     # print('not keep norm same')
                #     encoder_outputs = self.encoder.layer[i].output.LayerNorm(encoder_outputs)
                # print('get into joint sample copy')
                # encoder_outputs_new = (should_exit_wp_padded_tensor[:, i] * encoder_outputs) + (
                #             1 - should_exit_wp_padded_tensor[:, i]) * last_layer_hidden_state
                # last_layer_hidden_state = encoder_outputs_new
                # encoder_outputs = encoder_outputs_new
                # if self.args.keep_norm_same:
                #     # print('keep norm same')
                #     encoder_outputs = self.encoder.layer[i].output.LayerNorm(encoder_outputs)
                # pass

            else:
                res = []

                # if 'copy' in self.args.train_mode:
                #     last_layer_hidden_state = embedding_output

                    # raise NotImplementedError
                for i in range(self.config.num_hidden_layers):
                    encoder_outputs = self.encoder.adaptive_forward(
                        encoder_outputs, current_layer=i, attention_mask=extended_attention_mask, head_mask=head_mask
                    )

                    assert ptm_pool_pos is not None

                    if self.args.train_mode == 'joint_sample_copy_label':
                        raise NotImplementedError
                        # exit()
                        #随机sample label信息
                        pass
                    # pooled_output = self.pooler(encoder_outputs)
                    # logits = output_layers[i](output_dropout(pooled_output))
                    assert word_mask is not None
                    if self.args.train_mode == 'joint_sample_copy':
                        pass
                    else:
                        encoder_outputs = self.encoder.layer[i].output.LayerNorm(encoder_outputs)


                    if 'joint' in self.args.train_mode:
                        encoder_outputs_for_cls = batch_index_select_yf(encoder_outputs, ptm_pool_pos)
                        if extra_token_feature is not None:
                            encoder_outputs_for_cls = torch.cat([encoder_outputs_for_cls, extra_token_feature], dim=-1)
                        logits = output_layers[i](encoder_outputs_for_cls,word_mask)
                        res.append(logits)
                if self.args.train_mode == 'one_cls':
                    encoder_outputs_for_cls = batch_index_select_yf(encoder_outputs, ptm_pool_pos)
                    if extra_token_feature is not None:
                        encoder_outputs_for_cls = torch.cat([encoder_outputs_for_cls, extra_token_feature], dim=-1)
                    logits = output_layers[0](encoder_outputs_for_cls,word_mask)
                    assert len(res) == 0
                    res.append(logits)
        else:  # Use all layers for inference
            self.inference_instances_num += 1
            res = []
            # print('test_mode:{}'.format(self.args.test_mode))
            if self.args.test_mode == 'one_cls':
                # encoder_outputs = self.encoder(
                #     embedding_output,
                #     attention_mask=extended_attention_mask,
                #     head_mask=head_mask,
                #     # encoder_hidden_states=encoder_hidden_states,
                #     # encoder_attention_mask=encoder_ext ended_attention_mask,
                #     output_hidden_states=True
                # )

                for i in range(self.config.num_hidden_layers):
                    encoder_outputs = self.encoder.adaptive_forward(
                        encoder_outputs, current_layer=i, attention_mask=extended_attention_mask, head_mask=head_mask
                    )

                    encoder_outputs = self.encoder.layer[i].output.LayerNorm(encoder_outputs)



                    # pooled_output = self.pooler(encoder_outputs)
                    # logits = output_layers[i](output_dropout(pooled_output))

                    # encoder_outputs_for_cls = self.embed_dropout(encoder_outputs_for_cls)


                # for i in range(self.config.num_hidden_layers):
                #     encoder_outputs = self.encoder.adaptive_forward(
                #         encoder_outputs, current_layer=i, attention_mask=extended_attention_mask, head_mask=head_mask
                #     )
                #     print('{}:\n{}'.format(i, encoder_outputs[2][:2][:5]))

                    # assert ptm_pool_pos is not None
                # print('encoder_outputs:{}'.format(encoder_outputs[0][2][:2][:5]))
                #
                # hidden_states = encoder_outputs.hidden_states
                # print('hidden_states_num:{}'.format(len(hidden_states)))
                # for i, h in enumerate(hidden_states):
                #     print('{}:\n{}'.format(i, h[2][:2][:5]))
                # exit()

                # exit()

                assert ptm_pool_pos is not None
                encoder_outputs_for_cls = batch_index_select_yf(encoder_outputs, ptm_pool_pos)
                if extra_token_feature is not None:
                    encoder_outputs_for_cls = torch.cat([encoder_outputs_for_cls, extra_token_feature], dim=-1)

                logits = output_layers[-1](encoder_outputs_for_cls, word_mask)
                assert len(res) == 0
                res.append(logits)

            elif self.args.test_mode == 'joint':
                if 'label' in self.args.train_mode:
                    unk_label_embeddings = self.unk_label_embedding.unsqueeze(0).unsqueeze(0)
                # unk_label_embeddings = self.unk_label_embedding.unsqueeze(0).unsqueeze(0)
                for i in range(self.config.num_hidden_layers):
                    encoder_outputs = self.encoder.adaptive_forward(
                        encoder_outputs, current_layer=i, attention_mask=extended_attention_mask, head_mask=head_mask
                    )

                    encoder_outputs = self.encoder.layer[i].output.LayerNorm(encoder_outputs)

                    assert ptm_pool_pos is not None
                    encoder_outputs_for_cls = batch_index_select_yf(encoder_outputs, ptm_pool_pos)
                    if extra_token_feature is not None:
                        encoder_outputs_for_cls = torch.cat([encoder_outputs_for_cls, extra_token_feature], dim=-1)

                    # pooled_output = self.pooler(encoder_outputs)
                    # logits = output_layers[i](output_dropout(pooled_output))
                    assert word_mask is not None
                    # encoder_outputs_for_cls = self.embed_dropout(encoder_outputs_for_cls)
                    if 'label' in self.args.train_mode and self.args.true_label:
                        label_embeddings = unk_label_embeddings
                        label_embeddings = self.w_label_embedding(label_embeddings)
                    else:
                        label_embeddings = None
                    logits = output_layers[i](encoder_outputs_for_cls, word_mask,label_feature=label_embeddings)
                    res.append(logits)
            elif self.args.test_mode == 's_level_ee':
                # print(".test_mode == 's_level_ee'")
                # self.inference_instances_num += 1
                assert input_ids.size()[0] == 1
                for i in range(self.config.num_hidden_layers):
                    encoder_outputs = self.encoder.adaptive_forward(
                        encoder_outputs, current_layer=i, attention_mask=extended_attention_mask, head_mask=head_mask
                    )
                    # encoder_outputs = self.encoder.la
                    encoder_outputs = self.encoder.layer[i].output.LayerNorm(encoder_outputs)

                    assert ptm_pool_pos is not None
                    encoder_outputs_for_cls = batch_index_select_yf(encoder_outputs, ptm_pool_pos)
                    if extra_token_feature is not None:
                        encoder_outputs_for_cls = torch.cat([encoder_outputs_for_cls, extra_token_feature], dim=-1)

                    # pooled_output = self.pooler(encoder_outputs)
                    # logits = output_layers[i](output_dropout(pooled_output))
                    assert word_mask is not None
                    # encoder_outputs_for_cls = self.embed_dropout(encoder_outputs_for_cls)
                    logits = output_layers[i](encoder_outputs_for_cls, word_mask)

                    if (i+1) == self.config.num_hidden_layers:
                        self.exit_layer_num[i] += 1
                        self.inference_layers_num+=self.config.num_hidden_layers
                        # print('get last, exit!')
                        return [logits]

                    if self.args.criterion == 'entropy':
                        tmp_for_judging_ee = get_uncertainty(logits[0])

                        if self.args.sentence_ee_pool == 'avg':
                            tmp_for_judging_ee = torch.mean(tmp_for_judging_ee)
                            # pass
                        elif self.args.sentence_ee_pool == 'max':
                            tmp_for_judging_ee = torch.max(tmp_for_judging_ee)

                        if tmp_for_judging_ee<self.args.threshold:
                            self.inference_layers_num+=(i+1)
                            self.exit_layer_num[i]+=1
                            # print('early exit!')
                            return [logits]

                    elif self.args.criterion == 'max_p':
                        logits = nn.functional.softmax(logits,dim=-1)
                        tmp_for_judging_ee = torch.max(logits[0],dim=-1)[0]

                        if self.args.sentence_ee_pool == 'avg':
                            tmp_for_judging_ee = torch.mean(tmp_for_judging_ee)
                            # pass
                        elif self.args.sentence_ee_pool == 'max':
                            tmp_for_judging_ee = torch.min(tmp_for_judging_ee)

                        if tmp_for_judging_ee>self.args.threshold:
                            self.exit_layer_num[i] += 1
                            self.inference_layers_num+=(i+1)
                            # print('early exit!')
                            return [logits]


                    else:
                        raise NotImplementedError


                        # pass
                    # elif self.args.sentence_ee_pool == 'max':




                    # res.append(logits)
            elif self.args.test_mode == 't_level_ee':
                # print('input_ids:{}'.format(input_ids.size()))
                # print('ptm_pool_pos:{}'.format(ptm_pool_pos.size()))
                seq_len = ptm_pool_pos.size()[1]
                self.inference_token_num+=seq_len
                pred = [None] * seq_len
                exit_layer = [-1] * seq_len
                for i in range(self.config.num_hidden_layers):
                    encoder_outputs = self.encoder.adaptive_forward(
                        encoder_outputs, current_layer=i, attention_mask=extended_attention_mask, head_mask=head_mask
                    )
                    encoder_outputs = self.encoder.layer[i].output.LayerNorm(encoder_outputs)

                    assert ptm_pool_pos is not None
                    encoder_outputs_for_cls = batch_index_select_yf(encoder_outputs, ptm_pool_pos)
                    if extra_token_feature is not None:
                        encoder_outputs_for_cls = torch.cat([encoder_outputs_for_cls, extra_token_feature], dim=-1)

                    # pooled_output = self.pooler(encoder_outputs)
                    # logits = output_layers[i](output_dropout(pooled_output))
                    assert word_mask is not None
                    # encoder_outputs_for_cls = self.embed_dropout(encoder_outputs_for_cls)
                    logits = output_layers[i](encoder_outputs_for_cls, word_mask)
                    # print('logits:{}'.format(logits.size()))
                    if (i+1) == self.config.num_hidden_layers:
                        for j,_ in enumerate(pred):
                            if pred[j] is None:
                                self.exit_layer_num_token[i]+=1
                                pred[j] = logits[0,j]
                                exit_layer[j] = i
                        self.exit_layer_num[i]+=1
                        self.inference_layers_num+=(i+1)
                        return pred,exit_layer

                    # all_finished = True
                    if self.args.criterion == 'entropy':
                        tmp_for_judging_ee = get_uncertainty(logits[0])

                        for j, uncertainty in enumerate(tmp_for_judging_ee):
                            if pred[j] is None and uncertainty<self.args.threshold:
                                self.exit_layer_num_token[i] += 1
                                pred[j] = logits[0,j]
                                exit_layer[j] = i
                    elif self.args.criterion == 'max_p':

                        logits = nn.functional.softmax(logits,dim=-1)
                        for j,p in enumerate(logits[0]):
                            if pred[j] is None and p>self.args.threshold:
                                self.exit_layer_num_token[i] += 1
                                pred[j] = logits[0,j]
                                exit_layer[j] = i

                    all_finished = True

                    for p in pred:
                        if p is None:
                            all_finished = False
                            break

                    if all_finished:
                        self.exit_layer_num[i]+=1
                        self.inference_layers_num+=(i+1)
                        return pred,exit_layer
            elif self.args.test_mode == 't_level_win_ee':
                # print('input_ids:{}'.format(input_ids.size()))
                # print('ptm_pool_pos:{}'.format(ptm_pool_pos.size()))
                # mask_tensor = torch.zeros(size=1,)
                seq_len = ptm_pool_pos.size()[1]
                mask_tensor = torch.zeros(size=[1,seq_len,output_layers[-1].target_size],
                                          dtype=torch.float).to(ptm_pool_pos).float()

                self.inference_token_num+=seq_len
                pred = [None] * seq_len
                exit_layer = [-1] * seq_len
                for i in range(self.config.num_hidden_layers):
                    encoder_outputs = self.encoder.adaptive_forward(
                        encoder_outputs, current_layer=i, attention_mask=extended_attention_mask, head_mask=head_mask
                    )
                    encoder_outputs = self.encoder.layer[i].output.LayerNorm(encoder_outputs)

                    assert ptm_pool_pos is not None
                    encoder_outputs_for_cls = batch_index_select_yf(encoder_outputs, ptm_pool_pos)
                    if extra_token_feature is not None:
                        encoder_outputs_for_cls = torch.cat([encoder_outputs_for_cls, extra_token_feature], dim=-1)

                    # pooled_output = self.pooler(encoder_outputs)
                    # logits = output_layers[i](output_dropout(pooled_output))
                    assert word_mask is not None
                    # encoder_outputs_for_cls = self.embed_dropout(encoder_outputs_for_cls)
                    logits = output_layers[i](encoder_outputs_for_cls, word_mask)
                    if self.args.use_constrain and i>0:
                        logits+=mask_tensor
                        # logits = mask_logit_by_certain_pred_and_constrain(logits,pred,self.constrain_both)
                        pass
                    # print('logits:{}'.format(logits.size()))
                    if (i+1) == self.config.num_hidden_layers:
                        for j,_ in enumerate(pred):
                            if pred[j] is None:
                                self.exit_layer_num_token[i]+=1
                                pred[j] = logits[0,j]
                                exit_layer[j] = i
                        self.exit_layer_num[i]+=1
                        self.inference_layers_num+=(i+1)
                        # print('12layer exit, so inference_layers_num += 12')
                        return pred,exit_layer

                    # all_finished = True
                    if self.args.criterion == 'entropy':
                        tmp_for_judging_ee_token = get_uncertainty(logits[0])
                        # tmp_for_judging_ee_token =
                        tmp_for_judging_ee_token_uns = tmp_for_judging_ee_token.unsqueeze(0).unsqueeze(0)
                        tmp_for_judging_ee = nn.functional.max_pool1d(tmp_for_judging_ee_token_uns,kernel_size=self.args.win_size,stride=1,
                                                                      padding=self.args.win_size//2).squeeze(0).squeeze(0)
                        # print('tmp:{}'.format(tmp_for_judging_ee.size()))
                        # exit()

                        for j, uncertainty in enumerate(tmp_for_judging_ee):
                            if pred[j] is None and (uncertainty<self.args.threshold[i] or tmp_for_judging_ee_token[j]<self.args.t_level_t[i]):
                                self.exit_layer_num_token[i] += 1
                                pred[j] = logits[0,j]
                                if self.args.use_constrain:

                                    if j!=0:
                                        mask_tensor[0,j-1]+=self.constrain_both[1][torch.argmax(pred[j],dim=-1).item()]
                                    if j!=seq_len-1:
                                        mask_tensor[0,j+1]+=self.constrain_both[0][torch.argmax(pred[j],dim=-1).item()]
                                exit_layer[j] = i
                    elif self.args.criterion == 'max_p':
                        raise NotImplementedError
                        logits = nn.functional.softmax(logits,dim=-1)
                        for j,p in enumerate(logits[0]):
                            if pred[j] is None and p>self.args.threshold:
                                self.exit_layer_num_token[i] += 1
                                pred[j] = logits[0,j]
                                exit_layer[j] = i

                    all_finished = True

                    for p in pred:
                        if p is None:
                            all_finished = False
                            break

                    if all_finished:
                        self.exit_layer_num[i]+=1
                        self.inference_layers_num+=(i+1)
                        # print('{} layer exit, so inference_layers_num += {}'.format(i+1,i+1))
                        return pred,exit_layer

            elif self.args.test_mode == 't_level_win_ee_copy_pseudo':
                # print('input_ids:{}'.format(input_ids.size()))
                # print('ptm_pool_pos:{}'.format(ptm_pool_pos.size()))
                # mask_tensor = torch.zeros(size=1,)
                seq_len = ptm_pool_pos.size()[1]
                mask_tensor = torch.zeros(size=[1,seq_len,output_layers[-1].target_size],
                                          dtype=torch.float).to(ptm_pool_pos).float()
                wp_seq_len = input_ids.size(1)
                should_exit_wp_padded_tensor = torch.ones(size=[1, wp_seq_len, 1]).to(encoder_outputs)

                self.inference_token_num+=seq_len
                pred = [None] * seq_len
                exit_layer = [-1] * seq_len
                last_hidden_state = embedding_output
                last_hidden_state_before_layernorm = embedding_output
                for i in range(self.config.num_hidden_layers):
                    encoder_outputs = self.encoder.adaptive_forward(
                        last_hidden_state, current_layer=i, attention_mask=extended_attention_mask, head_mask=head_mask
                    )


                    # if not self.args.keep_norm_same:
                    #     encoder_outputs = self.encoder.layer[i].output.LayerNorm(encoder_outputs)
                    #
                    # last_hidden_state_maybe_before_layernorm = (should_exit_wp_padded_tensor * encoder_outputs) + (
                    #         1 - should_exit_wp_padded_tensor) * last_hidden_state_maybe_before_layernorm
                    #
                    # if self.args.keep_norm_same:
                    #     # print('keep norm same')
                    #     last_hidden_state = self.encoder.layer[i].output.LayerNorm(last_hidden_state_maybe_before_layernorm)
                    # else:
                    #     last_hidden_state = last_hidden_state_maybe_before_layernorm
                    if not self.args.keep_norm_same:
                        # print('not keep norm same')
                        encoder_outputs = self.encoder.layer[i].output.LayerNorm(encoder_outputs)
                        last_hidden_state = \
                            (should_exit_wp_padded_tensor * encoder_outputs) + \
                            (1 - should_exit_wp_padded_tensor) * last_hidden_state
                        encoder_outputs = last_hidden_state
                    else:
                        last_hidden_state_before_layernorm = \
                            (should_exit_wp_padded_tensor * encoder_outputs) + \
                            (1 - should_exit_wp_padded_tensor) * last_hidden_state_before_layernorm
                        last_hidden_state = self.encoder.layer[i].output.LayerNorm(last_hidden_state_before_layernorm)
                        encoder_outputs = last_hidden_state



                    assert ptm_pool_pos is not None
                    encoder_outputs_for_cls = batch_index_select_yf(encoder_outputs, ptm_pool_pos)
                    if extra_token_feature is not None:
                        encoder_outputs_for_cls = torch.cat([encoder_outputs_for_cls, extra_token_feature], dim=-1)

                    # pooled_output = self.pooler(encoder_outputs)
                    # logits = output_layers[i](output_dropout(pooled_output))
                    assert word_mask is not None
                    # encoder_outputs_for_cls = self.embed_dropout(encoder_outputs_for_cls)
                    logits = output_layers[i](encoder_outputs_for_cls, word_mask)
                    if self.args.use_constrain and i>0:
                        logits+=mask_tensor
                        # logits = mask_logit_by_certain_pred_and_constrain(logits,pred,self.constrain_both)
                        pass
                    # print('logits:{}'.format(logits.size()))
                    if (i+1) == self.config.num_hidden_layers:
                        for j,_ in enumerate(pred):
                            if pred[j] is None:
                                self.exit_layer_num_token[i]+=1
                                pred[j] = logits[0,j]
                                exit_layer[j] = i
                        self.exit_layer_num[i]+=1
                        self.inference_layers_num+=(i+1)
                        return pred,exit_layer

                    # all_finished = True
                    if self.args.criterion == 'entropy':
                        tmp_for_judging_ee_token = get_uncertainty(logits[0])
                        # tmp_for_judging_ee_token =
                        tmp_for_judging_ee_token_uns = tmp_for_judging_ee_token.unsqueeze(0).unsqueeze(0)
                        tmp_for_judging_ee = nn.functional.max_pool1d(tmp_for_judging_ee_token_uns,kernel_size=self.args.win_size,stride=1,
                                                                      padding=self.args.win_size//2).squeeze(0).squeeze(0)
                        # print('tmp:{}'.format(tmp_for_judging_ee.size()))
                        # exit()

                        for j, uncertainty in enumerate(tmp_for_judging_ee):
                            if pred[j] is None and (uncertainty<self.args.threshold[i] or tmp_for_judging_ee_token[j]<self.args.t_level_t[i]):
                                self.exit_layer_num_token[i] += 1
                                pred[j] = logits[0,j]
                                if self.args.true_copy:
                                    should_exit_wp_padded_tensor[0,j+1,0] = 0
                                if self.args.use_constrain:

                                    if j!=0:
                                        mask_tensor[0,j-1]+=self.constrain_both[1][torch.argmax(pred[j],dim=-1).item()]
                                    if j!=seq_len-1:
                                        mask_tensor[0,j+1]+=self.constrain_both[0][torch.argmax(pred[j],dim=-1).item()]
                                exit_layer[j] = i
                    elif self.args.criterion == 'max_p':
                        raise NotImplementedError
                        logits = nn.functional.softmax(logits,dim=-1)
                        for j,p in enumerate(logits[0]):
                            if pred[j] is None and p>self.args.threshold:
                                self.exit_layer_num_token[i] += 1
                                pred[j] = logits[0,j]
                                exit_layer[j] = i

                    all_finished = True

                    for p in pred:
                        if p is None:
                            all_finished = False
                            break

                    if all_finished:
                        self.exit_layer_num[i]+=1
                        self.inference_layers_num+=(i+1)
                        return pred,exit_layer

            elif self.args.test_mode == 't_level_win_ee_copy_pseudo_2':
                # print('input_ids:{}'.format(input_ids.size()))
                # print('ptm_pool_pos:{}'.format(ptm_pool_pos.size()))
                # mask_tensor = torch.zeros(size=1,)
                seq_len = ptm_pool_pos.size()[1]
                mask_tensor = torch.zeros(size=[1,seq_len,output_layers[-1].target_size],
                                          dtype=torch.float).to(ptm_pool_pos).float()
                wp_seq_len = input_ids.size(1)
                should_exit_wp_padded_tensor = torch.ones(size=[1, wp_seq_len, 1]).to(encoder_outputs)

                self.inference_token_num+=seq_len
                pred = [None] * seq_len
                exit_layer = [-1] * seq_len
                last_hidden_state = embedding_output
                last_hidden_state_before_layernorm = embedding_output
                for i in range(self.config.num_hidden_layers):
                    encoder_outputs = self.encoder.adaptive_forward(
                        last_hidden_state, current_layer=i, attention_mask=extended_attention_mask, head_mask=head_mask
                    )


                    # if not self.args.keep_norm_same:
                    #     encoder_outputs = self.encoder.layer[i].output.LayerNorm(encoder_outputs)
                    #
                    # last_hidden_state_maybe_before_layernorm = (should_exit_wp_padded_tensor * encoder_outputs) + (
                    #         1 - should_exit_wp_padded_tensor) * last_hidden_state_maybe_before_layernorm
                    #
                    # if self.args.keep_norm_same:
                    #     # print('keep norm same')
                    #     last_hidden_state = self.encoder.layer[i].output.LayerNorm(last_hidden_state_maybe_before_layernorm)
                    # else:
                    #     last_hidden_state = last_hidden_state_maybe_before_layernorm
                    if not self.args.keep_norm_same:
                        # print('not keep norm same')
                        encoder_outputs = self.encoder.layer[i].output.LayerNorm(encoder_outputs)
                        for j in range(should_exit_wp_padded_tensor.size(1)):
                            if should_exit_wp_padded_tensor[0, j, 0].item() == 1:
                                last_hidden_state[0,j] = encoder_outputs[0,j]
                        # last_hidden_state = \
                        #     (should_exit_wp_padded_tensor * encoder_outputs) + \
                        #     (1 - should_exit_wp_padded_tensor) * last_hidden_state
                        encoder_outputs = last_hidden_state
                    else:
                        # last_hidden_state_before_layernorm = \
                        #     (should_exit_wp_padded_tensor * encoder_outputs) + \
                        #     (1 - should_exit_wp_padded_tensor) * last_hidden_state_before_layernorm
                        for j in range(should_exit_wp_padded_tensor.size(1)):
                            if should_exit_wp_padded_tensor[0, j, 0].item() == 1:
                                last_hidden_state_before_layernorm[0,j] = encoder_outputs[0,j]
                        last_hidden_state = self.encoder.layer[i].output.LayerNorm(last_hidden_state_before_layernorm)
                        encoder_outputs = last_hidden_state



                    assert ptm_pool_pos is not None
                    encoder_outputs_for_cls = batch_index_select_yf(encoder_outputs, ptm_pool_pos)
                    if extra_token_feature is not None:
                        encoder_outputs_for_cls = torch.cat([encoder_outputs_for_cls, extra_token_feature], dim=-1)

                    # pooled_output = self.pooler(encoder_outputs)
                    # logits = output_layers[i](output_dropout(pooled_output))
                    assert word_mask is not None
                    # encoder_outputs_for_cls = self.embed_dropout(encoder_outputs_for_cls)
                    logits = output_layers[i](encoder_outputs_for_cls, word_mask)
                    if self.args.use_constrain and i>0:
                        logits+=mask_tensor
                        # logits = mask_logit_by_certain_pred_and_constrain(logits,pred,self.constrain_both)
                        pass
                    # print('logits:{}'.format(logits.size()))
                    if (i+1) == self.config.num_hidden_layers:
                        for j,_ in enumerate(pred):
                            if pred[j] is None:
                                self.exit_layer_num_token[i]+=1
                                pred[j] = logits[0,j]
                                exit_layer[j] = i
                        self.exit_layer_num[i]+=1
                        self.inference_layers_num+=(i+1)
                        return pred,exit_layer

                    # all_finished = True
                    if self.args.criterion == 'entropy':
                        tmp_for_judging_ee_token = get_uncertainty(logits[0])
                        # tmp_for_judging_ee_token =
                        tmp_for_judging_ee_token_uns = tmp_for_judging_ee_token.unsqueeze(0).unsqueeze(0)
                        tmp_for_judging_ee = nn.functional.max_pool1d(tmp_for_judging_ee_token_uns,kernel_size=self.args.win_size,stride=1,
                                                                      padding=self.args.win_size//2).squeeze(0).squeeze(0)
                        # print('tmp:{}'.format(tmp_for_judging_ee.size()))
                        # exit()

                        for j, uncertainty in enumerate(tmp_for_judging_ee):
                            if pred[j] is None and (uncertainty<self.args.threshold[i] or tmp_for_judging_ee_token[j]<self.args.t_level_t[i]):
                                self.exit_layer_num_token[i] += 1
                                pred[j] = logits[0,j]
                                if self.args.true_copy:
                                    should_exit_wp_padded_tensor[0,j+1,0] = 0
                                if self.args.use_constrain:

                                    if j!=0:
                                        mask_tensor[0,j-1]+=self.constrain_both[1][torch.argmax(pred[j],dim=-1).item()]
                                    if j!=seq_len-1:
                                        mask_tensor[0,j+1]+=self.constrain_both[0][torch.argmax(pred[j],dim=-1).item()]
                                exit_layer[j] = i
                    elif self.args.criterion == 'max_p':
                        raise NotImplementedError
                        logits = nn.functional.softmax(logits,dim=-1)
                        for j,p in enumerate(logits[0]):
                            if pred[j] is None and p>self.args.threshold:
                                self.exit_layer_num_token[i] += 1
                                pred[j] = logits[0,j]
                                exit_layer[j] = i

                    all_finished = True

                    for p in pred:
                        if p is None:
                            all_finished = False
                            break

                    if all_finished:
                        self.exit_layer_num[i]+=1
                        self.inference_layers_num+=(i+1)
                        return pred,exit_layer

            # elif self.args.test_mode == 't_level_win_ee_copy':
            #     # print('input_ids:{}'.format(input_ids.size()))
            #     # print('ptm_pool_pos:{}'.format(ptm_pool_pos.size()))
            #     # mask_tensor = torch.zeros(size=1,)
            #     seq_len = ptm_pool_pos.size()[1]
            #     mask_tensor = torch.zeros(size=[1,seq_len,output_layers[-1].target_size],
            #                               dtype=torch.float).to(ptm_pool_pos).float()
            #     wp_seq_len = input_ids.size(1)
            #     should_exit_wp_padded_tensor = torch.ones(size=[1, wp_seq_len, 1]).to(encoder_outputs)
            #
            #     self.inference_token_num+=seq_len
            #     pred = [None] * seq_len
            #     exit_layer = [-1] * seq_len
            #     last_hidden_state = embedding_output
            #     last_hidden_state_before_layernorm = embedding_output
            #     for i in range(self.config.num_hidden_layers):
            #
            #         should_update_indices = torch.nonzero(should_exit_wp_padded_tensor).squeeze(-1)
            #
            #         encoder_outputs = self.encoder.adaptive_forward(
            #             last_hidden_state, current_layer=i, attention_mask=extended_attention_mask, head_mask=head_mask,
            #             should_update_indices=should_update_indices
            #         )
            #
            #
            #         # if not self.args.keep_norm_same:
            #         #     encoder_outputs = self.encoder.layer[i].output.LayerNorm(encoder_outputs)
            #         #
            #         # last_hidden_state_maybe_before_layernorm = (should_exit_wp_padded_tensor * encoder_outputs) + (
            #         #         1 - should_exit_wp_padded_tensor) * last_hidden_state_maybe_before_layernorm
            #         #
            #         # if self.args.keep_norm_same:
            #         #     # print('keep norm same')
            #         #     last_hidden_state = self.encoder.layer[i].output.LayerNorm(last_hidden_state_maybe_before_layernorm)
            #         # else:
            #         #     last_hidden_state = last_hidden_state_maybe_before_layernorm
            #         if not self.args.keep_norm_same:
            #             # print('not keep norm same')
            #             encoder_outputs = self.encoder.layer[i].output.LayerNorm(encoder_outputs)
            #             # last_hidden_state = \
            #             #     (should_exit_wp_padded_tensor * encoder_outputs) + \
            #             #     (1 - should_exit_wp_padded_tensor) * last_hidden_state
            #             last_hidden_state[0,should_update_indices] = encoder_outputs
            #             encoder_outputs = last_hidden_state
            #         else:
            #             # last_hidden_state_before_layernorm = \
            #             #     (should_exit_wp_padded_tensor * encoder_outputs) + \
            #             #     (1 - should_exit_wp_padded_tensor) * last_hidden_state_before_layernorm
            #             last_hidden_state_before_layernorm[0,should_update_indices] = encoder_outputs
            #             last_hidden_state = self.encoder.layer[i].output.LayerNorm(last_hidden_state_before_layernorm)
            #             encoder_outputs = last_hidden_state
            #
            #
            #
            #         assert ptm_pool_pos is not None
            #         encoder_outputs_for_cls = batch_index_select_yf(encoder_outputs, ptm_pool_pos)
            #         if extra_token_feature is not None:
            #             encoder_outputs_for_cls = torch.cat([encoder_outputs_for_cls, extra_token_feature], dim=-1)
            #
            #         # pooled_output = self.pooler(encoder_outputs)
            #         # logits = output_layers[i](output_dropout(pooled_output))
            #         assert word_mask is not None
            #         # encoder_outputs_for_cls = self.embed_dropout(encoder_outputs_for_cls)
            #         logits = output_layers[i](encoder_outputs_for_cls, word_mask)
            #         if self.args.use_constrain and i>0:
            #             logits+=mask_tensor
            #             # logits = mask_logit_by_certain_pred_and_constrain(logits,pred,self.constrain_both)
            #             pass
            #         # print('logits:{}'.format(logits.size()))
            #         if (i+1) == self.config.num_hidden_layers:
            #             for j,_ in enumerate(pred):
            #                 if pred[j] is None:
            #                     self.exit_layer_num_token[i]+=1
            #                     pred[j] = logits[0,j]
            #                     exit_layer[j] = i
            #             self.exit_layer_num[i]+=1
            #             self.inference_layers_num+=(i+1)
            #             return pred,exit_layer
            #
            #         # all_finished = True
            #         if self.args.criterion == 'entropy':
            #             tmp_for_judging_ee_token = get_uncertainty(logits[0])
            #             # tmp_for_judging_ee_token =
            #             tmp_for_judging_ee_token_uns = tmp_for_judging_ee_token.unsqueeze(0).unsqueeze(0)
            #             tmp_for_judging_ee = nn.functional.max_pool1d(tmp_for_judging_ee_token_uns,kernel_size=self.args.win_size,stride=1,
            #                                                           padding=self.args.win_size//2).squeeze(0).squeeze(0)
            #             # print('tmp:{}'.format(tmp_for_judging_ee.size()))
            #             # exit()
            #
            #             for j, uncertainty in enumerate(tmp_for_judging_ee):
            #                 if pred[j] is None and (uncertainty<self.args.threshold[i] or tmp_for_judging_ee_token[j]<self.args.t_level_t[i]):
            #                     self.exit_layer_num_token[i] += 1
            #                     pred[j] = logits[0,j]
            #                     if self.args.true_copy:
            #                         should_exit_wp_padded_tensor[0,j+1,0] = 0
            #                     if self.args.use_constrain:
            #
            #                         if j!=0:
            #                             mask_tensor[0,j-1]+=self.constrain_both[1][torch.argmax(pred[j],dim=-1).item()]
            #                         if j!=seq_len-1:
            #                             mask_tensor[0,j+1]+=self.constrain_both[0][torch.argmax(pred[j],dim=-1).item()]
            #                     exit_layer[j] = i
            #         elif self.args.criterion == 'max_p':
            #             raise NotImplementedError
            #             logits = nn.functional.softmax(logits,dim=-1)
            #             for j,p in enumerate(logits[0]):
            #                 if pred[j] is None and p>self.args.threshold:
            #                     self.exit_layer_num_token[i] += 1
            #                     pred[j] = logits[0,j]
            #                     exit_layer[j] = i
            #
            #         all_finished = True
            #
            #         for p in pred:
            #             if p is None:
            #                 all_finished = False
            #                 break
            #
            #         if all_finished:
            #             self.exit_layer_num[i]+=1
            #             self.inference_layers_num+=(i+1)
            #             return pred,exit_layer

            elif self.args.test_mode == 't_level_win_ee_copy':
                seq_len = ptm_pool_pos.size()[1]
                now_ins_exit_layers = [11] * seq_len
                # print('input_ids:{}'.format(input_ids.size()))
                # print('ptm_pool_pos:{}'.format(ptm_pool_pos.size()))
                # mask_tensor = torch.zeros(size=1,)
                word_piece_num = kwargs['word_piece_num']
                first_word_pieces_pos = kwargs['first_word_pieces_pos']
                seq_len = ptm_pool_pos.size()[1]
                mask_tensor = torch.zeros(size=[1,seq_len,output_layers[-1].target_size],
                                          dtype=torch.float).to(ptm_pool_pos).float()
                wp_seq_len = input_ids.size(1)
                should_exit_wp_padded_tensor = torch.ones(size=[1, wp_seq_len, 1]).to(encoder_outputs)

                self.inference_token_num+=seq_len
                pred = [None] * seq_len
                exit_layer = [-1] * seq_len
                last_hidden_state = embedding_output
                last_hidden_state_before_layernorm = embedding_output
                for i in range(self.config.num_hidden_layers):

                    should_update_indices = torch.nonzero(should_exit_wp_padded_tensor[0,:,0]).squeeze(-1)

                    encoder_outputs = self.encoder.adaptive_forward(
                        last_hidden_state, current_layer=i, attention_mask=extended_attention_mask, head_mask=head_mask,
                        should_update_indices=should_update_indices
                    )


                    # if not self.args.keep_norm_same:
                    #     encoder_outputs = self.encoder.layer[i].output.LayerNorm(encoder_outputs)
                    #
                    # last_hidden_state_maybe_before_layernorm = (should_exit_wp_padded_tensor * encoder_outputs) + (
                    #         1 - should_exit_wp_padded_tensor) * last_hidden_state_maybe_before_layernorm
                    #
                    # if self.args.keep_norm_same:
                    #     # print('keep norm same')
                    #     last_hidden_state = self.encoder.layer[i].output.LayerNorm(last_hidden_state_maybe_before_layernorm)
                    # else:
                    #     last_hidden_state = last_hidden_state_maybe_before_layernorm
                    if not self.args.keep_norm_same:
                        # print('not keep norm same')
                        encoder_outputs = self.encoder.layer[i].output.LayerNorm(encoder_outputs)
                        # last_hidden_state = \
                        #     (should_exit_wp_padded_tensor * encoder_outputs) + \
                        #     (1 - should_exit_wp_padded_tensor) * last_hidden_state
                        # print('encoder_outputs:{}'.format(encoder_outputs.size()))
                        last_hidden_state[0,should_update_indices] = encoder_outputs
                        # last_hidden_state = encoder_outputs
                        encoder_outputs = last_hidden_state
                    else:
                        # last_hidden_state_before_layernorm = \
                        #     (should_exit_wp_padded_tensor * encoder_outputs) + \
                        #     (1 - should_exit_wp_padded_tensor) * last_hidden_state_before_layernorm
                        # print('encoder_outputs:{}'.format(encoder_outputs.size()))
                        # print('last_hidden_state_before_layernorm:{}'.format(last_hidden_state_before_layernorm.size()))
                        last_hidden_state_before_layernorm[0,should_update_indices] = encoder_outputs
                        # last_hidden_state_before_layernorm = encoder_outputs
                        last_hidden_state = self.encoder.layer[i].output.LayerNorm(last_hidden_state_before_layernorm)
                        encoder_outputs = last_hidden_state



                    assert ptm_pool_pos is not None
                    encoder_outputs_for_cls = batch_index_select_yf(encoder_outputs, ptm_pool_pos)
                    if extra_token_feature is not None:
                        encoder_outputs_for_cls = torch.cat([encoder_outputs_for_cls, extra_token_feature], dim=-1)

                    # pooled_output = self.pooler(encoder_outputs)
                    # logits = output_layers[i](output_dropout(pooled_output))
                    assert word_mask is not None
                    # encoder_outputs_for_cls = self.embed_dropout(encoder_outputs_for_cls)
                    logits = output_layers[i](encoder_outputs_for_cls, word_mask)
                    if self.args.use_constrain and i>0:
                        logits+=mask_tensor
                        # logits = mask_logit_by_certain_pred_and_constrain(logits,pred,self.constrain_both)
                        pass
                    # print('logits:{}'.format(logits.size()))
                    if (i+1) == self.config.max_inference_layer:
                        self.every_ins_token_exit_layers.append(now_ins_exit_layers)
                        if self.args.local_aware_2:
                            logits = nn.functional.softmax(logits,dim=-1)
                            logits[0] = filter_error_transitions(logits[0],pred,self.constrain_both[0],self.constrain_both[1])
                        for j,_ in enumerate(pred):
                            if pred[j] is None:
                                self.exit_layer_num_token[i]+=1
                                pred[j] = logits[0,j]
                                exit_layer[j] = i
                        self.exit_layer_num[i]+=1
                        self.inference_layers_num+=(i+1)
                        return pred,exit_layer

                    # all_finished = True
                    if self.args.criterion == 'entropy':
                        tmp_for_judging_ee_token = get_uncertainty(logits[0])
                        for j,p in enumerate(pred):
                            if p is not None:
                                tmp_for_judging_ee[j] = -1
                        # tmp_for_judging_ee_token =
                        tmp_for_judging_ee_token_uns = tmp_for_judging_ee_token.unsqueeze(0).unsqueeze(0)
                        # tmp_for_judging_ee_token
                        tmp_for_judging_ee = nn.functional.max_pool1d(tmp_for_judging_ee_token_uns,kernel_size=self.args.win_size,stride=1,
                                                                      padding=self.args.win_size//2).squeeze(0).squeeze(0)
                        # print('tmp:{}'.format(tmp_for_judging_ee.size()))
                        # exit()
                        now_layer_pred_labels = torch.argmax(logits[0],dim=-1)
                        for j, uncertainty in enumerate(tmp_for_judging_ee):
                            # print('self.args.local_aware_1:{}'.format(self.args.local_aware_1))
                            if self.args.local_aware_1:
                                # print(111)
                                if j>0:
                                    if pred[j-1] is not None:
                                        pass
                                    elif (now_layer_pred_labels[j-1].item(),now_layer_pred_labels[j].item()) in self.allowed_transitions:
                                        pass
                                    else:
                                        # print(11111111111111111111)
                                        continue
                                if j+1<seq_len:
                                    if pred[j+1] is not None:
                                        pass
                                    elif (now_layer_pred_labels[j].item(),now_layer_pred_labels[j+1].item()) in self.allowed_transitions:
                                        pass
                                    else:
                                        # print(22222222222222222222)
                                        continue
                            if pred[j] is None and (uncertainty<self.args.threshold[i] or tmp_for_judging_ee_token[j]<self.args.t_level_t[i]):
                                self.exit_layer_num_token[i] += 1
                                pred[j] = logits[0,j]
                                now_ins_exit_layers[j] = i
                                if self.args.true_copy:
                                    # should_exit_wp_padded_tensor[0,j+1,0] = 0
                                    if self.args.copy_wordpiece == 'all':
                                        should_exit_wp_padded_tensor[0,first_word_pieces_pos[0,j]:first_word_pieces_pos[0,j]+word_piece_num[0,j],0] = 0
                                    elif self.args.copy_wordpiece == 'first':
                                        should_exit_wp_padded_tensor[0,first_word_pieces_pos[0,j],0] = 0
                                if self.args.use_constrain:

                                    if j!=0:
                                        mask_tensor[0,j-1]+=self.constrain_both[1][torch.argmax(pred[j],dim=-1).item()]
                                    if j!=seq_len-1:
                                        mask_tensor[0,j+1]+=self.constrain_both[0][torch.argmax(pred[j],dim=-1).item()]
                                exit_layer[j] = i
                    elif self.args.criterion == 'max_p':
                        raise NotImplementedError
                        logits = nn.functional.softmax(logits,dim=-1)
                        for j,p in enumerate(logits[0]):
                            if pred[j] is None and p>self.args.threshold:
                                self.exit_layer_num_token[i] += 1
                                pred[j] = logits[0,j]
                                exit_layer[j] = i

                    all_finished = True

                    for p in pred:
                        if p is None:
                            all_finished = False
                            break

                    if all_finished:
                        self.exit_layer_num[i]+=1
                        self.inference_layers_num+=(i+1)
                        self.every_ins_token_exit_layers.append(now_ins_exit_layers)
                        return pred,exit_layer

            elif self.args.test_mode == 't_level_win_ee_copy_label':
                # print('input_ids:{}'.format(input_ids.size()))
                # print('ptm_pool_pos:{}'.format(ptm_pool_pos.size()))
                # mask_tensor = torch.zeros(size=1,)
                seq_len = ptm_pool_pos.size()[1]
                mask_tensor = torch.zeros(size=[1,seq_len,output_layers[-1].target_size],
                                          dtype=torch.float).to(ptm_pool_pos).float()
                wp_seq_len = input_ids.size(1)
                should_exit_wp_padded_tensor = torch.ones(size=[1, wp_seq_len, 1]).to(encoder_outputs)

                self.inference_token_num+=seq_len
                pred = [None] * seq_len
                exit_layer = [-1] * seq_len
                last_hidden_state = embedding_output
                last_hidden_state_before_layernorm = embedding_output
                known_label_embedding = self.unk_label_embedding.unsqueeze(0).unsqueeze(0).expand(1,seq_len,self.unk_label_embedding.size()[-1])
                for i in range(self.config.num_hidden_layers):

                    should_update_indices = torch.nonzero(should_exit_wp_padded_tensor[0,:,0]).squeeze(-1)

                    encoder_outputs = self.encoder.adaptive_forward(
                        last_hidden_state, current_layer=i, attention_mask=extended_attention_mask, head_mask=head_mask,
                        should_update_indices=should_update_indices
                    )


                    # if not self.args.keep_norm_same:
                    #     encoder_outputs = self.encoder.layer[i].output.LayerNorm(encoder_outputs)
                    #
                    # last_hidden_state_maybe_before_layernorm = (should_exit_wp_padded_tensor * encoder_outputs) + (
                    #         1 - should_exit_wp_padded_tensor) * last_hidden_state_maybe_before_layernorm
                    #
                    # if self.args.keep_norm_same:
                    #     # print('keep norm same')
                    #     last_hidden_state = self.encoder.layer[i].output.LayerNorm(last_hidden_state_maybe_before_layernorm)
                    # else:
                    #     last_hidden_state = last_hidden_state_maybe_before_layernorm
                    if not self.args.keep_norm_same:
                        # print('not keep norm same')
                        encoder_outputs = self.encoder.layer[i].output.LayerNorm(encoder_outputs)
                        # last_hidden_state = \
                        #     (should_exit_wp_padded_tensor * encoder_outputs) + \
                        #     (1 - should_exit_wp_padded_tensor) * last_hidden_state
                        # print('encoder_outputs:{}'.format(encoder_outputs.size()))
                        last_hidden_state[0,should_update_indices] = encoder_outputs
                        # last_hidden_state = encoder_outputs
                        encoder_outputs = last_hidden_state
                    else:
                        # last_hidden_state_before_layernorm = \
                        #     (should_exit_wp_padded_tensor * encoder_outputs) + \
                        #     (1 - should_exit_wp_padded_tensor) * last_hidden_state_before_layernorm
                        # print('encoder_outputs:{}'.format(encoder_outputs.size()))
                        # print('last_hidden_state_before_layernorm:{}'.format(last_hidden_state_before_layernorm.size()))
                        last_hidden_state_before_layernorm[0,should_update_indices] = encoder_outputs
                        # last_hidden_state_before_layernorm = encoder_outputs
                        last_hidden_state = self.encoder.layer[i].output.LayerNorm(last_hidden_state_before_layernorm)
                        encoder_outputs = last_hidden_state



                    assert ptm_pool_pos is not None
                    encoder_outputs_for_cls = batch_index_select_yf(encoder_outputs, ptm_pool_pos)
                    if extra_token_feature is not None:
                        encoder_outputs_for_cls = torch.cat([encoder_outputs_for_cls, extra_token_feature], dim=-1)

                    # pooled_output = self.pooler(encoder_outputs)
                    # logits = output_layers[i](output_dropout(pooled_output))
                    assert word_mask is not None
                    # encoder_outputs_for_cls = self.embed_dropout(encoder_outputs_for_cls)

                    known_label_embedding_for_cls = nn.functional.linear(known_label_embedding, self.w_label_embedding.weight, self.w_label_embedding.bias)

                    logits = output_layers[i](encoder_outputs_for_cls, word_mask,label_feature=known_label_embedding_for_cls)
                    if self.args.use_constrain and i>0:
                        logits+=mask_tensor
                        # logits = mask_logit_by_certain_pred_and_constrain(logits,pred,self.constrain_both)
                        pass
                    # print('logits:{}'.format(logits.size()))
                    if (i+1) == self.config.max_inference_layer:
                        if self.args.local_aware_2:
                            logits = nn.functional.softmax(logits,dim=-1)
                            logits[0] = filter_error_transitions(logits[0],pred,self.constrain_both[0],self.constrain_both[1])
                        for j,_ in enumerate(pred):
                            if pred[j] is None:
                                self.exit_layer_num_token[i]+=1
                                pred[j] = logits[0,j]
                                exit_layer[j] = i
                        self.exit_layer_num[i]+=1
                        self.inference_layers_num+=(i+1)
                        return pred,exit_layer

                    # all_finished = True
                    if self.args.criterion == 'entropy':
                        tmp_for_judging_ee_token = get_uncertainty(logits[0])
                        for j,p in enumerate(pred):
                            if p is not None:
                                tmp_for_judging_ee[j] = -1
                        # tmp_for_judging_ee_token =
                        tmp_for_judging_ee_token_uns = tmp_for_judging_ee_token.unsqueeze(0).unsqueeze(0)
                        # tmp_for_judging_ee_token
                        tmp_for_judging_ee = nn.functional.max_pool1d(tmp_for_judging_ee_token_uns,kernel_size=self.args.win_size,stride=1,
                                                                      padding=self.args.win_size//2).squeeze(0).squeeze(0)
                        # print('tmp:{}'.format(tmp_for_judging_ee.size()))
                        # exit()
                        now_layer_pred_labels = torch.argmax(logits[0],dim=-1)
                        for j, uncertainty in enumerate(tmp_for_judging_ee):
                            # print('self.args.local_aware_1:{}'.format(self.args.local_aware_1))
                            if self.args.local_aware_1:
                                # print(111)
                                if j>0:
                                    if pred[j-1] is not None:
                                        pass
                                    elif (now_layer_pred_labels[j-1].item(),now_layer_pred_labels[j].item()) in self.allowed_transitions:
                                        pass
                                    else:
                                        # print(11111111111111111111)
                                        continue
                                if j+1<seq_len:
                                    if pred[j+1] is not None:
                                        pass
                                    elif (now_layer_pred_labels[j].item(),now_layer_pred_labels[j+1].item()) in self.allowed_transitions:
                                        pass
                                    else:
                                        # print(22222222222222222222)
                                        continue
                            if pred[j] is None and (uncertainty<self.args.threshold[i] or tmp_for_judging_ee_token[j]<self.args.t_level_t[i]):
                                self.exit_layer_num_token[i] += 1
                                pred[j] = logits[0,j]
                                if self.args.true_label:
                                    known_label_embedding[0,j] = nn.functional.embedding(torch.argmax(pred[j],dim=-1),self.label_embedding_weight,
                                                                                         None,None,2.,False,False)
                                if self.args.true_copy:
                                    should_exit_wp_padded_tensor[0,j+1,0] = 0
                                if self.args.use_constrain:

                                    if j!=0:
                                        mask_tensor[0,j-1]+=self.constrain_both[1][torch.argmax(pred[j],dim=-1).item()]
                                    if j!=seq_len-1:
                                        mask_tensor[0,j+1]+=self.constrain_both[0][torch.argmax(pred[j],dim=-1).item()]
                                exit_layer[j] = i
                    elif self.args.criterion == 'max_p':
                        raise NotImplementedError
                        logits = nn.functional.softmax(logits,dim=-1)
                        for j,p in enumerate(logits[0]):
                            if pred[j] is None and p>self.args.threshold:
                                self.exit_layer_num_token[i] += 1
                                pred[j] = logits[0,j]
                                exit_layer[j] = i

                    all_finished = True

                    for p in pred:
                        if p is None:
                            all_finished = False
                            break

                    if all_finished:
                        self.exit_layer_num[i]+=1
                        self.inference_layers_num+=(i+1)
                        return pred,exit_layer

            # elif self.args.test_mode == 't_level_win_ee_copy':
            #     #请注意layernorm的正确使用
            #     # raise NotImplementedError
            #     # print('input_ids:{}'.format(input_ids.size()))
            #     # print('ptm_pool_pos:{}'.format(ptm_pool_pos.size()))
            #     # mask_tensor = torch.zeros(size=1,)
            #     seq_len = ptm_pool_pos.size()[1]
            #     mask_tensor = torch.zeros(size=[1,seq_len,output_layers[-1].target_size],
            #                               dtype=torch.float).to(ptm_pool_pos).float()
            #     wp_seq_len = input_ids.size(1)
            #     should_exit_wp_padded_tensor = torch.ones(size=[1,wp_seq_len,1],dtype=torch.long).to(encoder_outputs)
            #     # should_exit_wp_padded_tensor[0,0] = 0
            #     self.inference_token_num+=seq_len
            #     pred = [None] * seq_len
            #     exit_layer = [-1] * seq_len
            #     last_layer_hidden_state = embedding_output
            #     history_encoder_outputs = []
            #     for i in range(self.config.num_hidden_layers):
            #         if self.args.true_copy:
            #             should_update_indices = torch.nonzero(should_exit_wp_padded_tensor).squeeze(-1)
            #         else:
            #             should_update_indices = None
            #         encoder_outputs = self.encoder.adaptive_forward(
            #             last_layer_hidden_state, current_layer=i, attention_mask=extended_attention_mask, head_mask=head_mask,
            #             should_update_indices=should_update_indices
            #         )
            #
            #         if not self.args.keep_norm_same:
            #             # print('not keep norm same')
            #             encoder_outputs = self.encoder.layer[i].output.LayerNorm(encoder_outputs)
            #         # print(222)
            #         history_encoder_outputs.append(encoder_outputs)
            #
            #         if self.args.true_copy:
            #             last_layer_hidden_state[0,should_update_indices] = encoder_outputs
            #             # print('true copy')
            #             # print(torch.mean(should_exit_wp_padded_tensor).item())
            #             # print()
            #
            #             # for j in range(should_exit_wp_padded_tensor.size(1)):
            #             #     if should_exit_wp_padded_tensor[0,j,0].item() == 1:
            #             #         last_layer_hidden_state[0,j] = encoder_outputs[0,j]
            #
            #             # should_update_indices = torch.nonzero(should_exit_wp_padded_tensor).squeeze(-1)
            #
            #         else:
            #             last_layer_hidden_state = encoder_outputs
            #             # encoder_outputs_new = (should_exit_wp_padded_tensor * encoder_outputs) + (1-should_exit_wp_padded_tensor) * last_layer_hidden_state
            #             # last_layer_hidden_state = encoder_outputs_new
            #             # encoder_outputs = encoder_outputs_new
            #         encoder_outputs = last_layer_hidden_state
            #         if self.args.keep_norm_same:
            #             # print('keep norm same')
            #             encoder_outputs = self.encoder.layer[i].output.LayerNorm(encoder_outputs)
            #             if self.args.true_copy:
            #                 last_layer_hidden_state = encoder_outputs
            #
            #         assert ptm_pool_pos is not None
            #         encoder_outputs_for_cls = batch_index_select_yf(encoder_outputs, ptm_pool_pos)
            #         if extra_token_feature is not None:
            #             encoder_outputs_for_cls = torch.cat([encoder_outputs_for_cls, extra_token_feature], dim=-1)
            #
            #         # pooled_output = self.pooler(encoder_outputs)
            #         # logits = output_layers[i](output_dropout(pooled_output))
            #         assert word_mask is not None
            #         # encoder_outputs_for_cls = self.embed_dropout(encoder_outputs_for_cls)
            #         logits = output_layers[i](encoder_outputs_for_cls, word_mask)
            #         if self.args.use_constrain and i>0:
            #             logits+=mask_tensor
            #             # logits = mask_logit_by_certain_pred_and_constrain(logits,pred,self.constrain_both)
            #             pass
            #         # print('logits:{}'.format(logits.size()))
            #         if (i+1) == self.config.num_hidden_layers:
            #             for j,_ in enumerate(pred):
            #                 if pred[j] is None:
            #                     self.exit_layer_num_token[i]+=1
            #                     pred[j] = logits[0,j]
            #                     exit_layer[j] = i
            #             self.exit_layer_num[i]+=1
            #             self.inference_layers_num+=(i+1)
            #             break
            #             return pred,exit_layer
            #
            #         # all_finished = True
            #         if self.args.criterion == 'entropy':
            #             tmp_for_judging_ee_token = get_uncertainty(logits[0])
            #             # tmp_for_judging_ee_token =
            #             tmp_for_judging_ee_token_uns = tmp_for_judging_ee_token.unsqueeze(0).unsqueeze(0)
            #             tmp_for_judging_ee = nn.functional.max_pool1d(tmp_for_judging_ee_token_uns,kernel_size=self.args.win_size,stride=1,
            #                                                           padding=self.args.win_size//2).squeeze(0).squeeze(0)
            #             # print('tmp:{}'.format(tmp_for_judging_ee.size()))
            #             # exit()
            #
            #             for j, uncertainty in enumerate(tmp_for_judging_ee):
            #                 if pred[j] is None and (uncertainty<self.args.threshold[i] or tmp_for_judging_ee_token[j]<self.args.t_level_t[i]):
            #                     self.exit_layer_num_token[i] += 1
            #                     pred[j] = logits[0,j]
            #                     #暂时只支持中文
            #                     # should_exit_wp_padded_tensor = 0
            #                     # should_exit_wp_padded_tensor[5:] = 0
            #
            #                     should_exit_wp_padded_tensor[0,j+1,0] = 0
            #                     # should_exit_wp_padded_tensor[j+1] = 0
            #
            #                     if self.args.use_constrain:
            #
            #                         if j!=0:
            #                             mask_tensor[0,j-1]+=self.constrain_both[1][torch.argmax(pred[j],dim=-1).item()]
            #                         if j!=seq_len-1:
            #                             mask_tensor[0,j+1]+=self.constrain_both[0][torch.argmax(pred[j],dim=-1).item()]
            #                     exit_layer[j] = i
            #         elif self.args.criterion == 'max_p':
            #             raise NotImplementedError
            #             logits = nn.functional.softmax(logits,dim=-1)
            #             for j,p in enumerate(logits[0]):
            #                 if pred[j] is None and p>self.args.threshold:
            #                     self.exit_layer_num_token[i] += 1
            #                     pred[j] = logits[0,j]
            #                     exit_layer[j] = i
            #
            #         all_finished = True
            #
            #         for p in pred:
            #             if p is None:
            #                 all_finished = False
            #                 break
            #
            #         if all_finished:
            #             self.exit_layer_num[i]+=1
            #             self.inference_layers_num+=(i+1)
            #             break
            #             return pred,exit_layer
            #     #请注意layernorm的正确使用
            #     # raise NotImplementedErrorshou
            #     # for j in range(seq_len):
            #     #
            #     #     # print(111)
            #     #     now_token_exit_layer = exit_layer[j]
            #     #     now_token_true_hidden = history_encoder_outputs[now_token_exit_layer][0,j+1]
            #     #     same_tmp = torch.mean((now_token_true_hidden == last_layer_hidden_state[0,j+1]).float())
            #     #     assert(same_tmp>0.999)
            #
            #     return pred,exit_layer
            #
            # elif self.args.test_mode == 't_level_win_ee_copy_pseudo':
            #     # print('input_ids:{}'.format(input_ids.size()))
            #     # print('ptm_pool_pos:{}'.format(ptm_pool_pos.size()))
            #     # mask_tensor = torch.zeros(size=1,)
            #     seq_len = ptm_pool_pos.size()[1]
            #     mask_tensor = torch.zeros(size=[1,seq_len,output_layers[-1].target_size],
            #                               dtype=torch.float).to(ptm_pool_pos).float()
            #     wp_seq_len = input_ids.size(1)
            #     should_exit_wp_padded_tensor = torch.ones(size=[1,wp_seq_len,1]).to(encoder_outputs)
            #     # should_exit_wp_padded_tensor[0,0] = 0
            #     self.inference_token_num+=seq_len
            #     pred = [None] * seq_len
            #     exit_layer = [-1] * seq_len
            #     last_layer_hidden_state = 0
            #     for i in range(self.config.num_hidden_layers):
            #         encoder_outputs = self.encoder.adaptive_forward(
            #             encoder_outputs, current_layer=i, attention_mask=extended_attention_mask, head_mask=head_mask
            #         )
            #
            #         if not self.args.keep_norm_same:
            #             # print('not keep norm same')
            #             encoder_outputs = self.encoder.layer[i].output.LayerNorm(encoder_outputs)
            #         if self.args.true_copy:
            #             # print('true copy')
            #             # print(torch.mean(should_exit_wp_padded_tensor).item())
            #             # print()
            #             encoder_outputs_new = (should_exit_wp_padded_tensor * encoder_outputs) + (1-should_exit_wp_padded_tensor) * last_layer_hidden_state
            #             last_layer_hidden_state = encoder_outputs_new
            #             encoder_outputs = encoder_outputs_new
            #
            #         if self.args.keep_norm_same:
            #             # print('keep norm same')
            #             encoder_outputs = self.encoder.layer[i].output.LayerNorm(encoder_outputs)
            #
            #         assert ptm_pool_pos is not None
            #         encoder_outputs_for_cls = batch_index_select_yf(encoder_outputs, ptm_pool_pos)
            #         if extra_token_feature is not None:
            #             encoder_outputs_for_cls = torch.cat([encoder_outputs_for_cls, extra_token_feature], dim=-1)
            #
            #         # pooled_output = self.pooler(encoder_outputs)
            #         # logits = output_layers[i](output_dropout(pooled_output))
            #         assert word_mask is not None
            #         # encoder_outputs_for_cls = self.embed_dropout(encoder_outputs_for_cls)
            #         logits = output_layers[i](encoder_outputs_for_cls, word_mask)
            #         if self.args.use_constrain and i>0:
            #             logits+=mask_tensor
            #             # logits = mask_logit_by_certain_pred_and_constrain(logits,pred,self.constrain_both)
            #             pass
            #         # print('logits:{}'.format(logits.size()))
            #         if (i+1) == self.config.num_hidden_layers:
            #             for j,_ in enumerate(pred):
            #                 if pred[j] is None:
            #                     self.exit_layer_num_token[i]+=1
            #                     pred[j] = logits[0,j]
            #                     exit_layer[j] = i
            #             self.exit_layer_num[i]+=1
            #             self.inference_layers_num+=(i+1)
            #             return pred,exit_layer
            #
            #         # all_finished = True
            #         if self.args.criterion == 'entropy':
            #             tmp_for_judging_ee_token = get_uncertainty(logits[0])
            #             # tmp_for_judging_ee_token =
            #             tmp_for_judging_ee_token_uns = tmp_for_judging_ee_token.unsqueeze(0).unsqueeze(0)
            #             tmp_for_judging_ee = nn.functional.max_pool1d(tmp_for_judging_ee_token_uns,kernel_size=self.args.win_size,stride=1,
            #                                                           padding=self.args.win_size//2).squeeze(0).squeeze(0)
            #             # print('tmp:{}'.format(tmp_for_judging_ee.size()))
            #             # exit()
            #
            #             for j, uncertainty in enumerate(tmp_for_judging_ee):
            #                 if pred[j] is None and (uncertainty<self.args.threshold[i] or tmp_for_judging_ee_token[j]<self.args.t_level_t[i]):
            #                     self.exit_layer_num_token[i] += 1
            #                     pred[j] = logits[0,j]
            #                     #暂时只支持中文
            #                     # should_exit_wp_padded_tensor = 0
            #                     # should_exit_wp_padded_tensor[5:] = 0
            #
            #                     should_exit_wp_padded_tensor[0,j+1,0] = 0
            #                     # should_exit_wp_padded_tensor[j+1] = 0
            #
            #                     if self.args.use_constrain:
            #
            #                         if j!=0:
            #                             mask_tensor[0,j-1]+=self.constrain_both[1][torch.argmax(pred[j],dim=-1).item()]
            #                         if j!=seq_len-1:
            #                             mask_tensor[0,j+1]+=self.constrain_both[0][torch.argmax(pred[j],dim=-1).item()]
            #                     exit_layer[j] = i
            #         elif self.args.criterion == 'max_p':
            #             raise NotImplementedError
            #             logits = nn.functional.softmax(logits,dim=-1)
            #             for j,p in enumerate(logits[0]):
            #                 if pred[j] is None and p>self.args.threshold:
            #                     self.exit_layer_num_token[i] += 1
            #                     pred[j] = logits[0,j]
            #                     exit_layer[j] = i
            #
            #         all_finished = True
            #
            #         for p in pred:
            #             if p is None:
            #                 all_finished = False
            #                 break
            #
            #         if all_finished:
            #             self.exit_layer_num[i]+=1
            #             self.inference_layers_num+=(i+1)
            #             return pred,exit_layer
            #     #请注意layernorm的正确使用
            #     # raise NotImplementedErrorshou
            #
            # elif self.args.test_mode == 't_level_win_ee_copy_pseudo_2':
            #     # print('input_ids:{}'.format(input_ids.size()))
            #     # print('ptm_pool_pos:{}'.format(ptm_pool_pos.size()))
            #     # mask_tensor = torch.zeros(size=1,)
            #     seq_len = ptm_pool_pos.size()[1]
            #     mask_tensor = torch.zeros(size=[1,seq_len,output_layers[-1].target_size],
            #                               dtype=torch.float).to(ptm_pool_pos).float()
            #     wp_seq_len = input_ids.size(1)
            #     should_exit_wp_padded_tensor = torch.ones(size=[1,wp_seq_len,1],dtype=torch.long).to(encoder_outputs)
            #     # should_exit_wp_padded_tensor[0,0] = 0
            #     self.inference_token_num+=seq_len
            #     pred = [None] * seq_len
            #     exit_layer = [-1] * seq_len
            #     last_layer_hidden_state = embedding_output
            #     history_encoder_outputs = []
            #     for i in range(self.config.num_hidden_layers):
            #         encoder_outputs = self.encoder.adaptive_forward(
            #             last_layer_hidden_state, current_layer=i, attention_mask=extended_attention_mask, head_mask=head_mask
            #         )
            #
            #         if not self.args.keep_norm_same:
            #             # print('not keep norm same')
            #             encoder_outputs = self.encoder.layer[i].output.LayerNorm(encoder_outputs)
            #         # print(222)
            #         history_encoder_outputs.append(encoder_outputs)
            #
            #         if self.args.true_copy:
            #             # print('true copy')
            #             # print(torch.mean(should_exit_wp_padded_tensor).item())
            #             # print()
            #             for j in range(should_exit_wp_padded_tensor.size(1)):
            #                 if should_exit_wp_padded_tensor[0,j,0].item() == 1:
            #                     last_layer_hidden_state[0,j] = encoder_outputs[0,j]
            #         else:
            #             last_layer_hidden_state = encoder_outputs
            #             # encoder_outputs_new = (should_exit_wp_padded_tensor * encoder_outputs) + (1-should_exit_wp_padded_tensor) * last_layer_hidden_state
            #             # last_layer_hidden_state = encoder_outputs_new
            #             # encoder_outputs = encoder_outputs_new
            #         encoder_outputs = last_layer_hidden_state
            #         if self.args.keep_norm_same:
            #             # print('keep norm same')
            #             encoder_outputs = self.encoder.layer[i].output.LayerNorm(encoder_outputs)
            #             if self.args.true_copy:
            #                 last_layer_hidden_state = encoder_outputs
            #
            #         assert ptm_pool_pos is not None
            #         encoder_outputs_for_cls = batch_index_select_yf(encoder_outputs, ptm_pool_pos)
            #         if extra_token_feature is not None:
            #             encoder_outputs_for_cls = torch.cat([encoder_outputs_for_cls, extra_token_feature], dim=-1)
            #
            #         # pooled_output = self.pooler(encoder_outputs)
            #         # logits = output_layers[i](output_dropout(pooled_output))
            #         assert word_mask is not None
            #         # encoder_outputs_for_cls = self.embed_dropout(encoder_outputs_for_cls)
            #         logits = output_layers[i](encoder_outputs_for_cls, word_mask)
            #         if self.args.use_constrain and i>0:
            #             logits+=mask_tensor
            #             # logits = mask_logit_by_certain_pred_and_constrain(logits,pred,self.constrain_both)
            #             pass
            #         # print('logits:{}'.format(logits.size()))
            #         if (i+1) == self.config.num_hidden_layers:
            #             for j,_ in enumerate(pred):
            #                 if pred[j] is None:
            #                     self.exit_layer_num_token[i]+=1
            #                     pred[j] = logits[0,j]
            #                     exit_layer[j] = i
            #             self.exit_layer_num[i]+=1
            #             self.inference_layers_num+=(i+1)
            #             break
            #             return pred,exit_layer
            #
            #         # all_finished = True
            #         if self.args.criterion == 'entropy':
            #             tmp_for_judging_ee_token = get_uncertainty(logits[0])
            #             # tmp_for_judging_ee_token =
            #             tmp_for_judging_ee_token_uns = tmp_for_judging_ee_token.unsqueeze(0).unsqueeze(0)
            #             tmp_for_judging_ee = nn.functional.max_pool1d(tmp_for_judging_ee_token_uns,kernel_size=self.args.win_size,stride=1,
            #                                                           padding=self.args.win_size//2).squeeze(0).squeeze(0)
            #             # print('tmp:{}'.format(tmp_for_judging_ee.size()))
            #             # exit()
            #
            #             for j, uncertainty in enumerate(tmp_for_judging_ee):
            #                 if pred[j] is None and (uncertainty<self.args.threshold[i] or tmp_for_judging_ee_token[j]<self.args.t_level_t[i]):
            #                     self.exit_layer_num_token[i] += 1
            #                     pred[j] = logits[0,j]
            #                     #暂时只支持中文
            #                     # should_exit_wp_padded_tensor = 0
            #                     # should_exit_wp_padded_tensor[5:] = 0
            #
            #                     should_exit_wp_padded_tensor[0,j+1,0] = 0
            #                     # should_exit_wp_padded_tensor[j+1] = 0
            #
            #                     if self.args.use_constrain:
            #
            #                         if j!=0:
            #                             mask_tensor[0,j-1]+=self.constrain_both[1][torch.argmax(pred[j],dim=-1).item()]
            #                         if j!=seq_len-1:
            #                             mask_tensor[0,j+1]+=self.constrain_both[0][torch.argmax(pred[j],dim=-1).item()]
            #                     exit_layer[j] = i
            #         elif self.args.criterion == 'max_p':
            #             raise NotImplementedError
            #             logits = nn.functional.softmax(logits,dim=-1)
            #             for j,p in enumerate(logits[0]):
            #                 if pred[j] is None and p>self.args.threshold:
            #                     self.exit_layer_num_token[i] += 1
            #                     pred[j] = logits[0,j]
            #                     exit_layer[j] = i
            #
            #         all_finished = True
            #
            #         for p in pred:
            #             if p is None:
            #                 all_finished = False
            #                 break
            #
            #         if all_finished:
            #             self.exit_layer_num[i]+=1
            #             self.inference_layers_num+=(i+1)
            #             break
            #             return pred,exit_layer
            #     #请注意layernorm的正确使用
            #     # raise NotImplementedErrorshou
            #     # for j in range(seq_len):
            #     #
            #     #     # print(111)
            #     #     now_token_exit_layer = exit_layer[j]
            #     #     now_token_true_hidden = history_encoder_outputs[now_token_exit_layer][0,j+1]
            #     #     same_tmp = torch.mean((now_token_true_hidden == last_layer_hidden_state[0,j+1]).float())
            #     #     assert(same_tmp>0.999)
            #
            #     return pred,exit_layer




            else:
                raise NotImplementedError
            # pooled_output = self.pooler(encoder_outputs[0])
            # res = [output_layers[self.config.num_hidden_layers - 1](pooled_output)]
        # else:
        #     patient_counter = 0
        #     patient_result = None
        #     calculated_layer_num = 0
        #     for i in range(self.config.num_hidden_layers):
        #         calculated_layer_num += 1
        #         encoder_outputs = self.encoder.adaptive_forward(
        #             encoder_outputs, current_layer=i, attention_mask=extended_attention_mask, head_mask=head_mask
        #         )
        #
        #         pooled_output = self.pooler(encoder_outputs)
        #         logits = output_layers[i](pooled_output)
        #         if regression:
        #             labels = logits.detach()
        #             if patient_result is not None:
        #                 patient_labels = patient_result.detach()
        #             if (patient_result is not None) and torch.abs(patient_result - labels) < self.regression_threshold:
        #                 patient_counter += 1
        #             else:
        #                 patient_counter = 0
        #         else:
        #             labels = logits.detach().argmax(dim=1)
        #             if patient_result is not None:
        #                 patient_labels = patient_result.detach().argmax(dim=1)
        #             if (patient_result is not None) and torch.all(labels.eq(patient_labels)):
        #                 patient_counter += 1
        #             else:
        #                 patient_counter = 0
        #
        #         patient_result = logits
        #         if patient_counter == self.patience:
        #             break
        #     res = [patient_result]
        #     self.inference_layers_num += calculated_layer_num
        #     self.inference_instances_num += 1

        return res

class BERT_SeqLabel_EE(nn.Module):
    def __init__(self,bundle,args,):
        super().__init__()
        if hasattr(bundle,'embeddings'):
            if 'bigram' in bundle.embeddings:
                self.bigram_embedding = bundle.embeddings['bigram']
        else:
            self.bigram_embedding = None
        if args.language_ == 'cn':
            args.use_char = 0
        if args.use_char and args.language_ == 'en':
            self.cnn_char = CNNCharEmbedding(bundle.vocabs['words'],word_dropout=0.01)

        if args.use_word:
            if args.language_ == 'cn':
                self.word_embedding = bundle.embeddings['char']
                print('{}\nuse word!\n{}'.format('*'*20,'*'*20))
            else:
                if args.word_embed_dim == 100:
                    self.word_embedding = bundle.embeddings['word_100']
                elif args.word_embed_dim == 300:
                    self.word_embedding = bundle.embeddings['word_300']
                else:
                    raise NotImplementedError

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
                raise NotImplementedError
                self.ptm_config = AlbertConfig.from_pretrained(args.ptm_name)
                self.ptm_encoder = AlbertModel.from_pretrained(args.ptm_name)
            elif 'roberta' in args.ptm_name and args.language_ == 'en':
                raise NotImplementedError
            elif 'roberta' in args.ptm_name and args.language_ == 'cn':
                self.ptm_config = BertConfig.from_pretrained(args.ptm_name)
                self.ptm_encoder = BertModel_EE.from_pretrained(args.ptm_name,args=args,output_hidden_states=True)
            elif 'bert' in args.ptm_name:
                self.ptm_config = BertConfig.from_pretrained(args.ptm_name)
                self.ptm_encoder = BertModel_EE.from_pretrained(args.ptm_name,args=args,output_hidden_states=True)
                # raise NotImplementedError
            else:
                raise NotImplementedError
            # print('ptm_model:{}'.format(self.ptm_encoder))
            self.ptm_encoder.resize_token_embeddings(new_num_tokens=len(bundle.tokenizer))

        self.ptm_config.keep_norm_same = args.keep_norm_same
        self.ptm_config.train_mode = args.train_mode
        self.ptm_config.test_mode = args.train_mode
        self.layer_exit_pred = []
        for i in range(self.ptm_config.num_hidden_layers):
            self.layer_exit_pred.append([])
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
        w_dim_inp_size = self.ptm_config.hidden_size
        if args.after_bert == 'tener':
            w_dim_inp_size = self.ptm_config.hidden_size
            if self.args.language_ == 'cn' and self.args.use_bigram and self.bigram_embedding:
                w_dim_inp_size = self.ptm_config.hidden_size + self.bigram_embedding._embed_size
            if self.args.use_char:
                w_dim_inp_size = self.ptm_config.hidden_size + self.cnn_char._embed_size

            if self.args.use_word:
                w_dim_inp_size += self.word_embedding._embed_size

            if 'label' in self.args.train_mode:
                w_dim_inp_size+=args.cls_hidden*args.cls_ff


            if 'joint' in args.train_mode:
                self.classfiers = nn.ModuleList([
                    TENER_classifier(args,w_dim_inp_size,target_size) for _ in range(self.ptm_config.num_hidden_layers)])
            elif args.train_mode == 'one_cls':
                self.classfiers = nn.ModuleList([TENER_classifier(args,w_dim_inp_size,target_size)])


            # self.w_dim = nn.Linear(w_dim_inp_size, args.cls_hidden)
            # self.transformer_cls = TransformerEncoder(1,args.cls_hidden,args.cls_head,args.cls_ff*args.cls_hidden,dropout=args.cls_dropout,
            #                                      after_norm=args.cls_after_norm,attn_type='adatrans',scale=args.cls_scale,dropout_attn=args.cls_drop_attn,
            #                                      pos_embed=None,target_size=target_size)
            # self.cls_out = nn.Sequential([
            #     nn.Linear(self.ptm_config.hidden_size,args.cls_hidden),
            #     transformer_cls,
            # ])
            # self.cls_out = lambda x,mask: self.transformer_cls(self.w_dim(x),mask)
        elif args.after_bert == 'linear':
            # raise NotImplementedError
            # if args.use_fastnlp_bert:
            #     self.cls_out = nn.Linear(768, target_size)
            # else:
            #     self.cls_out = nn.Linear(self.ptm_config.hidden_size,target_size)

            if 'joint' in args.train_mode:
                self.classfiers = nn.ModuleList([
                    Pseudo_Classifier(args,w_dim_inp_size,target_size) for _ in range(self.ptm_config.num_hidden_layers)])
            elif args.train_mode == 'one_cls':
                self.classfiers = nn.ModuleList([Pseudo_Classifier(args,w_dim_inp_size,target_size)])

        if self.use_crf:
            # raise NotImplementedError
            self.crf = get_crf_zero_init(target_size,True)


        # self.embed_dropout = nn.Dropout(args.embed_dropout)



    def forward(self,**kwargs):
        # print('V3')
        # print(kwargs.keys())
        words = kwargs['words']
        seq_len = kwargs['seq_len']
        target = kwargs['target']

        word_pieces = kwargs['word_pieces']
        attention_mask = kwargs['bert_attention_mask']
        # print('words:{}'.format(words[0]))
        # print('word_pieces:{}'.format(word_pieces[0]))

        ptm_pool_pos = None
        if self.args.ptm_pool_method == 'first':
            ptm_pool_pos = kwargs['first_word_pieces_pos']
        elif self.args.ptm_pool_method == 'first_skip_space':
            ptm_pool_pos = kwargs['first_word_pieces_pos_skip_space']
        elif self.args.ptm_pool_method == 'last':
            ptm_pool_pos = kwargs['last_word_pieces_pos']


        if self.args.use_fastnlp_bert:
            raise NotImplementedError
            ptm_output = self.ptm_encoder(words)
            encoded = ptm_output
        else:
            word_mask = seq_len_to_mask(seq_len)
            extra_features = None
            if self.args.use_word:
                extra_features = self.word_embedding(words)
            if self.args.language_ == 'cn' and self.args.use_bigram and self.bigram_embedding:
                bigrams = kwargs['bigrams']
                bigrams = self.bigram_embedding(bigrams)
                if extra_features is None:
                    extra_features = bigrams
                else:
                    extra_features = torch.cat([extra_features,bigrams],dim=-1)
                # encoded = torch.cat([encoded,bigrams],dim=-1)
            if self.args.use_char:
                char_encoded = self.cnn_char(words)
                if extra_features is None:
                    extra_features = char_encoded
                else:
                    extra_features = torch.cat([extra_features,char_encoded],dim=-1)
                # encoded = torch.cat([encoded,char_encoded],dim=-1)

            if self.training and (self.args.train_mode == 'joint_sample_copy' or self.args.train_mode=='joint_sample_copy_label'):
                should_exit_word_padded_tensor = kwargs['should_exit_word_padded_tensor']
                should_exit_wp_padded_tensor = kwargs['should_exit_wp_padded_tensor']
            else:
                should_exit_word_padded_tensor = None
                should_exit_wp_padded_tensor = None


            gold_labels = None

            ptm_output = self.ptm_encoder(input_ids=word_pieces,attention_mask=attention_mask,token_type_ids=None,position_ids=None,
                                          output_layers=self.classfiers,ptm_pool_pos=ptm_pool_pos,extra_token_feature=extra_features,word_mask=word_mask,
                                          should_exit_word_padded_tensor=should_exit_word_padded_tensor,should_exit_wp_padded_tensor=should_exit_wp_padded_tensor,
                                          gold_labels=gold_labels,
                                          first_word_pieces_pos=kwargs.get('first_word_pieces_pos'),
                                          word_piece_num=kwargs.get('word_piece_num')
                                          )

            result = {}
            batch_size = words.size(0)
            max_seq_len = words.size(1)
            if self.training:
                if self.args.train_mode == 'joint_sample_copy_label' or (self.args.train_mode == 'joint_sample_copy' and self.args.copy_supervised == 0):
                    assert self.args.train_mode != 'joint_two_stage'
                    # print(1)
                    loss = 0
                    assert len(ptm_output) == self.ptm_config.num_hidden_layers
                    # target = target.flatten(0, 1)
                    if self.args.joint_weighted:
                        total_weight = 0
                        for i, pred in enumerate(ptm_output):

                            if self.use_crf:
                                mask = seq_len_to_mask(seq_len)
                                now_layer_loss = self.crf(pred, target, mask).mean(dim=0)
                            else:
                                pred = pred.flatten(0, 1)
                                if self.args.true_label:
                                    target_now_layer = target.masked_fill(mask=~(should_exit_word_padded_tensor[:,i]),value=-100).flatten(0,1)
                                else:
                                    target_now_layer = target.flatten(0, 1)
                                now_layer_loss = self.loss_func(pred, target_now_layer)
                                if self.loss_func.reduction == 'none':
                                    if self.args.flooding:
                                        # print('flooding!')
                                        now_layer_loss_bias = kwargs['flooding_bias'][:,i]
                                        now_layer_loss = now_layer_loss.view(batch_size,max_seq_len)
                                        # print('flooding_bias:{}'.format(self.args.flooding_bias))
                                        now_layer_loss = torch.abs(now_layer_loss-now_layer_loss_bias-self.args.flooding_bias)
                                        # print(now_layer_loss)
                                        # now_layer_loss = torch.mean(now_layer_loss)
                                        now_layer_loss = now_layer_loss.view(*target.size())

                                    non_pad_target_num = torch.sum((target != -100).float())
                                    now_layer_loss = torch.sum(now_layer_loss)/non_pad_target_num

                                # now_layer_loss_extra = nn.functional.cross_entropy(pred,target,ignore_index=-100,reduction='mean')

                            loss += now_layer_loss * (i+1)
                            total_weight += (i+1)
                        loss/=total_weight

                    else:
                        for i,pred in enumerate(ptm_output):

                            if self.use_crf:
                                mask = seq_len_to_mask(seq_len)
                                loss += self.crf(pred, target, mask).mean(dim=0)
                            else:
                                pred = pred.flatten(0, 1)

                                loss += self.loss_func(pred, target)

                        loss/=len(ptm_output)
                    result['loss'] = loss
                if self.args.train_mode == 'joint_two_stage':
                    loss = 0
                    assert len(ptm_output) == self.ptm_config.num_hidden_layers
                    target = target.flatten(0, 1)
                    if self.args.joint_weighted:
                        raise NotImplementedError
                        total_weight = 0
                        for i, pred in enumerate(ptm_output):

                            if self.use_crf:
                                mask = seq_len_to_mask(seq_len)
                                now_layer_loss = self.crf(pred, target, mask).mean(dim=0)
                            else:
                                pred = pred.flatten(0, 1)

                                now_layer_loss = self.loss_func(pred, target)

                                # now_layer_loss_extra = nn.functional.cross_entropy(pred,target,ignore_index=-100,reduction='mean')

                            loss += now_layer_loss * (i+1)
                            total_weight += (i+1)
                        # loss/=total_weight

                    else:
                        for i,pred in enumerate(ptm_output[:-1]):

                            if self.use_crf:
                                mask = seq_len_to_mask(seq_len)
                                loss += self.crf(pred, target, mask).mean(dim=0)
                            else:
                                pred = pred.flatten(0, 1)

                                loss += self.loss_func(pred, target)

                        # loss/=len(ptm_output)
                    result['loss'] = loss
                elif 'joint' in self.args.train_mode:
                    loss = 0
                    assert len(ptm_output) == self.ptm_config.num_hidden_layers
                    target = target.flatten(0, 1)
                    if self.args.joint_weighted:
                        total_weight = 0
                        for i, pred in enumerate(ptm_output):

                            if self.use_crf:
                                mask = seq_len_to_mask(seq_len)
                                now_layer_loss = self.crf(pred, target, mask).mean(dim=0)
                            else:
                                pred = pred.flatten(0, 1)

                                now_layer_loss = self.loss_func(pred, target)
                                if self.loss_func.reduction == 'none':
                                    if self.args.flooding:
                                        # print('flooding!')
                                        now_layer_loss_bias = kwargs['flooding_bias'][:,i]
                                        now_layer_loss = now_layer_loss.view(batch_size,max_seq_len)
                                        # now_layer_loss = torch.abs(now_layer_loss-now_layer_loss_bias)
                                        now_layer_loss = torch.abs(
                                            now_layer_loss - now_layer_loss_bias - self.args.flooding_bias)
                                        # now_layer_loss = torch.mean(now_layer_loss)
                                        now_layer_loss = now_layer_loss.view(*target.size())

                                    non_pad_target_num = torch.sum((target != -100).float())
                                    now_layer_loss = torch.sum(now_layer_loss)/non_pad_target_num

                                # now_layer_loss_extra = nn.functional.cross_entropy(pred,target,ignore_index=-100,reduction='mean')

                            loss += now_layer_loss * (i+1)
                            total_weight += (i+1)
                        loss/=total_weight

                    else:
                        for i,pred in enumerate(ptm_output):

                            if self.use_crf:
                                mask = seq_len_to_mask(seq_len)
                                loss += self.crf(pred, target, mask).mean(dim=0)
                            else:
                                pred = pred.flatten(0, 1)

                                loss += self.loss_func(pred, target)

                        loss/=len(ptm_output)
                    result['loss'] = loss

                elif self.args.train_mode == 'one_cls':
                    pred = ptm_output[0]
                    if self.use_crf:
                        mask = seq_len_to_mask(seq_len)
                        loss = self.crf(pred, target, mask).mean(dim=0)
                    else:
                        pred = pred.flatten(0, 1)
                        target = target.flatten(0, 1)
                        loss = self.loss_func(pred, target)

                    result['loss'] = loss
            else:
                if self.args.test_mode == 'joint':
                    # ptm_output = torch.cat(ptm_output,dim=0)
                    ptm_output = torch.stack(ptm_output,dim=0)
                    result['pred'] = ptm_output
                elif self.args.test_mode == 'one_cls':
                    # return ptm_output
                    ptm_output = torch.stack(ptm_output,dim=0)
                    result['pred'] = ptm_output
                elif self.args.test_mode == 's_level_ee':
                    result['pred'] = torch.stack(ptm_output,dim=0)
                    # result['pred']
                    # print(ptm_output[0].size())
                    # print(result['pred'].size())
                    # exit()
                elif self.args.test_mode == 't_level_ee':
                    assert words.size(0) == 1
                    ptm_output,exit_layer = ptm_output
                    assert -1 not in exit_layer
                    # print(ptm_output)
                    # print(ptm_output[0].size())
                    # print(len(ptm_output))
                    ptm_output = torch.stack(ptm_output,dim=0).unsqueeze(0).unsqueeze(0)
                    pred_tags = torch.argmax(ptm_output,dim=-1)
                    for k in range(len(exit_layer)):
                        self.layer_exit_pred[exit_layer[k]].append(pred_tags[0][0][k].item())
                    # print(ptm_output.size())

                    result['pred'] = ptm_output
                    # exit()
                elif self.args.test_mode == 't_level_win_ee':
                    ptm_output,exit_layer = ptm_output
                    assert -1 not in exit_layer
                    assert words.size(0) == 1
                    ptm_output = torch.stack(ptm_output,dim=0).unsqueeze(0).unsqueeze(0)
                    pred_tags = torch.argmax(ptm_output, dim=-1)
                    # print(ptm_output.size())
                    for k in range(len(exit_layer)):
                        self.layer_exit_pred[exit_layer[k]].append(pred_tags[0][0][k].item())
                    result['pred'] = ptm_output

                elif self.args.test_mode == 't_level_win_ee_copy_pseudo':
                    ptm_output,exit_layer = ptm_output
                    assert -1 not in exit_layer
                    assert words.size(0) == 1
                    ptm_output = torch.stack(ptm_output,dim=0).unsqueeze(0).unsqueeze(0)
                    pred_tags = torch.argmax(ptm_output, dim=-1)
                    # print(ptm_output.size())
                    for k in range(len(exit_layer)):
                        self.layer_exit_pred[exit_layer[k]].append(pred_tags[0][0][k].item())
                    result['pred'] = ptm_output
                elif self.args.test_mode == 't_level_win_ee_copy_pseudo_2':
                    ptm_output,exit_layer = ptm_output
                    assert -1 not in exit_layer
                    assert words.size(0) == 1
                    ptm_output = torch.stack(ptm_output,dim=0).unsqueeze(0).unsqueeze(0)
                    pred_tags = torch.argmax(ptm_output, dim=-1)
                    # print(ptm_output.size())
                    for k in range(len(exit_layer)):
                        self.layer_exit_pred[exit_layer[k]].append(pred_tags[0][0][k].item())
                    result['pred'] = ptm_output
                elif self.args.test_mode == 't_level_win_ee_copy':
                    ptm_output,exit_layer = ptm_output
                    assert -1 not in exit_layer
                    assert words.size(0) == 1
                    ptm_output = torch.stack(ptm_output,dim=0).unsqueeze(0).unsqueeze(0)
                    pred_tags = torch.argmax(ptm_output, dim=-1)
                    # print(ptm_output.size())
                    for k in range(len(exit_layer)):
                        self.layer_exit_pred[exit_layer[k]].append(pred_tags[0][0][k].item())
                    result['pred'] = ptm_output
                elif self.args.test_mode == 't_level_win_ee_copy_2':
                    ptm_output,exit_layer = ptm_output
                    assert -1 not in exit_layer
                    assert words.size(0) == 1
                    ptm_output = torch.stack(ptm_output,dim=0).unsqueeze(0).unsqueeze(0)
                    pred_tags = torch.argmax(ptm_output, dim=-1)
                    # print(ptm_output.size())
                    for k in range(len(exit_layer)):
                        self.layer_exit_pred[exit_layer[k]].append(pred_tags[0][0][k].item())
                    result['pred'] = ptm_output
                elif self.args.test_mode == 't_level_win_ee_copy_label':
                    ptm_output,exit_layer = ptm_output
                    assert -1 not in exit_layer
                    assert words.size(0) == 1
                    ptm_output = torch.stack(ptm_output,dim=0).unsqueeze(0).unsqueeze(0)
                    pred_tags = torch.argmax(ptm_output, dim=-1)
                    # print(ptm_output.size())
                    for k in range(len(exit_layer)):
                        self.layer_exit_pred[exit_layer[k]].append(pred_tags[0][0][k].item())
                    result['pred'] = ptm_output
                else:
                    raise  NotImplementedError


        # print('encoded:{}'.format(type(encoded)))
        # print('encoded:{}'.format(encoded))
        # print('last_hidden_state:{}'.format(encoded.last_hidden_state.size()))
        # print('pooler_output:{}'.format(encoded.pooler_output.size()))
        # ptm_output = encoded
        #     encoded = ptm_output.last_hidden_state
        #     encoded = batch_index_select_yf(encoded, ptm_pool_pos)

        # print('word_pieces:{}'.format(word_pieces))
        # print('encoded:{}\n{}'.format(encoded.size(),encoded[0][:5][:5]))
        # exit()

        # if self.bigram_embedding:




        # if self.args.after_bert == 'tener':
        #     # print('encoded:{}'.format(encoded.size()))
        #     # print('attn_mask:{}'.format(attention_mask.size()))
        #     if self.args.use_bigram and self.bigram_embedding:
        #         bigrams = kwargs['bigrams']
        #         bigrams = self.bigram_embedding(bigrams)
        #         encoded = torch.cat([encoded,bigrams],dim=-1)
        #     if self.args.use_char:
        #         char_encoded = self.cnn_char(words)
        #         encoded = torch.cat([encoded,char_encoded],dim=-1)
        #     encoded = self.embed_dropout(encoded)
        #     tener_attn_mask = seq_len_to_mask(seq_len)
        #
        #     pred = self.cls_out(encoded,tener_attn_mask)
        # elif self.args.after_bert == 'linear':
        #     pred = self.cls_out(encoded)



        # encoded = batch_index_select_yf(encoded, ptm_pool_pos)
        # print('pred:{}'.format(pred.size()))
        # print('ptm_pool_pos:{}'.format(ptm_pool_pos))
        # pred = batch_index_select_yf(pred,ptm_pool_pos)


        # result = {}
        #
        # if self.use_crf:
        #     mask = seq_len_to_mask(seq_len)
        #     if self.training:
        #         loss = self.crf(pred,target,mask).mean(dim=0)
        #         result['loss'] = loss
        #     else:
        #         pred, path = self.crf.viterbi_decode(pred,mask)
        #         result['pred'] = pred
        #
        # else:
        #
        #
        #     if self.training:
        #         pred = pred.flatten(0,1)
        #         target = target.flatten(0,1)
        #         loss = self.loss_func(pred,target)
        #         result['loss'] = loss
        #     else:
        #         result['pred'] = pred
        return result





