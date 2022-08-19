from fastNLP import seq_len_to_mask
from fastNLP.modules import ConditionalRandomField
from utils import MyDropout
from transformers import AutoModel,AutoConfig,AutoTokenizer,AlbertTokenizerFast,AlbertTokenizer,BertConfig,BertModel,BertTokenizer
from transformers.models.bert.modeling_bert import BertLayer,BertEncoder,BertModel,BertAttention,BertSelfAttention,BertOutput
# from transformers.models.albert.modeling_albert import
from modules import TransformerEncoder
from utils import batch_index_select_yf
from fastNLP.embeddings import CNNCharEmbedding
from transformers.modeling_utils import apply_chunking_to_forward
from utils import get_uncertainty,get_entropy
from utils import mask_logit_by_certain_pred_and_constrain
from utils import filter_error_transitions
import torch.nn as nn
from V0.modeling_bert_ee import Pseudo_Classifier
import torch
from transformers.models.albert.modeling_albert import AlbertModel,AlbertTransformer,AlbertLayer,AlbertLayerGroup,AlbertAttention,AlbertConfig
from transformers.activations import ACT2FN
import math
from transformers.modeling_outputs import BaseModelOutputWithPooling

class AlbertAttention_EE_pseudo_counter(nn.Module):
    def forward(self,input_ids,
                attention_mask=None,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                output_attentions=False):
        pass


#相比原本的AlbertAttention，多支持了像BertAttention一样的encoder decoder attention模式
class AlbertAttention_EE(AlbertAttention):

    def __init__(self,config):
        super().__init__(config)
        self.counter = AlbertAttention_EE_pseudo_counter()

    # Copied from transformers.models.bert.modeling_bert.BertSelfAttention.transpose_for_scores

    def forward(self, input_ids,
                attention_mask=None,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                output_attentions=False):

        mixed_query_layer = self.query(input_ids)

        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(input_ids)
            mixed_value_layer = self.value(input_ids)

        self.counter(mixed_query_layer, None, None, mixed_key_layer, None, None)


        # mixed_key_layer = self.key(input_ids)
        # mixed_value_layer = self.value(input_ids)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attention_dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        # Should find a better way to do this
        w = (
            self.dense.weight.t()
            .view(self.num_attention_heads, self.attention_head_size, self.hidden_size)
            .to(context_layer.dtype)
        )
        b = self.dense.bias.to(context_layer.dtype)

        projected_context_layer = torch.einsum("bfnd,ndh->bfh", context_layer, w) + b
        projected_context_layer_dropout = self.output_dropout(projected_context_layer)
        layernormed_context_layer = self.LayerNorm(input_ids + projected_context_layer_dropout)
        return (layernormed_context_layer, attention_probs) if output_attentions else (layernormed_context_layer,)

class AlbertLayer_EE(AlbertLayer):
    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.full_layer_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = AlbertAttention_EE(config)
        self.ffn = nn.Linear(config.hidden_size, config.intermediate_size)
        self.ffn_output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = ACT2FN[config.hidden_act]
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, output_hidden_states=False,should_update_indices=None
    ):

        q_hidden_states = hidden_states
        k_v_hidden_states = hidden_states

        if should_update_indices is not None:
            assert q_hidden_states.size(0) == 1
            q_hidden_states = q_hidden_states[:,should_update_indices]


        # attention_output = self.attention(hidden_states, attention_mask, head_mask, output_attentions)
        attention_output = self.attention(
            q_hidden_states,
            # attention_mask,
            None,
            head_mask,
            k_v_hidden_states,
            attention_mask,
            output_attentions,
        )

        ffn_output = apply_chunking_to_forward(
            self.ff_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output[0],
        )
        hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])

        return (hidden_states,) + attention_output[1:]  # add attentions if we output them

    # def ff_chunk(self, attention_output):
    #     ffn_output = self.ffn(attention_output)
    #     ffn_output = self.activation(ffn_output)
    #     ffn_output = self.ffn_output(ffn_output)
    #     return ffn_output

class AlbertLayerGroup_EE(AlbertLayerGroup):
    def __init__(self, config):
        super().__init__(config)

        self.albert_layers = nn.ModuleList([AlbertLayer_EE(config) for _ in range(config.inner_group_num)])

    def forward(
        self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, output_hidden_states=False,should_update_indices=None,
    ):
        layer_hidden_states = ()
        layer_attentions = ()

        for layer_index, albert_layer in enumerate(self.albert_layers):
            layer_output = albert_layer(hidden_states, attention_mask, head_mask[layer_index], output_attentions,should_update_indices=should_update_indices)
            hidden_states = layer_output[0]

            if output_attentions:
                layer_attentions = layer_attentions + (layer_output[1],)

            if output_hidden_states:
                layer_hidden_states = layer_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if output_hidden_states:
            outputs = outputs + (layer_hidden_states,)
        if output_attentions:
            outputs = outputs + (layer_attentions,)
        return outputs  # last-layer hidden state, (layer hidden states), (layer attentions)



class AlbertTransformer_EE(AlbertTransformer):

    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.embedding_hidden_mapping_in = nn.Linear(config.embedding_size, config.hidden_size)
        self.albert_layer_groups = nn.ModuleList([AlbertLayerGroup_EE(config) for _ in range(config.num_hidden_groups)])

    def adaptive_forward(self, hidden_states, current_layer, attention_mask=None, head_mask=None, should_update_indices=None):
        if current_layer == 0:
            hidden_states = hidden_states
            # hidden_states = self.embedding_hidden_mapping_in(hidden_states)
        else:
            hidden_states = hidden_states

        layers_per_group = int(self.config.num_hidden_layers / self.config.num_hidden_groups)

        # Index of the hidden group
        group_idx = int(current_layer / (self.config.num_hidden_layers / self.config.num_hidden_groups))

        layer_group_output = self.albert_layer_groups[group_idx](
            hidden_states,
            attention_mask,
            head_mask[group_idx * layers_per_group : (group_idx + 1) * layers_per_group],
            should_update_indices=should_update_indices
        )
        hidden_states = layer_group_output[0]

        return hidden_states

class AlbertModel_EE(AlbertModel):
    def __init__(self, config, args, add_pooling_layer=True, ):
        super().__init__(config,add_pooling_layer=True)
        self.args = args
        self.encoder = AlbertTransformer_EE(config)
        self.init_weights()
        self.reset_stats()

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

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        output_layers=None,
        ptm_pool_pos=None,
        extra_token_feature=None,
        word_mask=None,
        should_exit_word_padded_tensor=None,
        should_exit_wp_padded_tensor=None,
        return_dict=None,
        **kwargs
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        # embedding_output = self.

        encoder_outputs = embedding_output

        encoder_outputs = self.encoder.embedding_hidden_mapping_in(encoder_outputs)
        embedding_output = encoder_outputs

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
                        # encoder_outputs = self.encoder.layer[i].output.LayerNorm(encoder_outputs)
                        last_hidden_state = \
                            (should_exit_wp_padded_tensor[:,i] * encoder_outputs) + \
                            (1 - should_exit_wp_padded_tensor[:,i]) * last_hidden_state
                        encoder_outputs = last_hidden_state
                    else:
                        last_hidden_state_before_layernorm = \
                            (should_exit_wp_padded_tensor[:,i] * encoder_outputs) + \
                            (1 - should_exit_wp_padded_tensor[:,i]) * last_hidden_state_before_layernorm
                        last_hidden_state = last_hidden_state_before_layernorm
                        # last_hidden_state = self.encoder.layer[i].output.LayerNorm(last_hidden_state_before_layernorm)
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
                        pass
                        # encoder_outputs = self.encoder.layer[i].output.LayerNorm(encoder_outputs)


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

                    # encoder_outputs = self.encoder.layer[i].output.LayerNorm(encoder_outputs)



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
                # return encoder_outputs
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

                    # encoder_outputs = self.encoder.layer[i].output.LayerNorm(encoder_outputs)

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
                    # encoder_outputs = self.encoder.layer[i].output.LayerNorm(encoder_outputs)

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
                    # encoder_outputs = self.encoder.layer[i].output.LayerNorm(encoder_outputs)

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
                    # encoder_outputs = self.encoder.layer[i].output.LayerNorm(encoder_outputs)

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
                        # encoder_outputs = self.encoder.layer[i].output.LayerNorm(encoder_outputs)
                        last_hidden_state = \
                            (should_exit_wp_padded_tensor * encoder_outputs) + \
                            (1 - should_exit_wp_padded_tensor) * last_hidden_state
                        encoder_outputs = last_hidden_state
                    else:
                        last_hidden_state_before_layernorm = \
                            (should_exit_wp_padded_tensor * encoder_outputs) + \
                            (1 - should_exit_wp_padded_tensor) * last_hidden_state_before_layernorm
                        last_hidden_state = last_hidden_state_before_layernorm
                        # last_hidden_state = self.encoder.layer[i].output.LayerNorm(last_hidden_state_before_layernorm)
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
                        # encoder_outputs = self.encoder.layer[i].output.LayerNorm(encoder_outputs)
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
                        last_hidden_state = last_hidden_state_before_layernorm
                        # last_hidden_state = self.encoder.layer[i].output.LayerNorm(last_hidden_state_before_layernorm)
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
                        # encoder_outputs = self.encoder.layer[i].output.LayerNorm(encoder_outputs)
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
                        last_hidden_state = last_hidden_state_before_layernorm
                        # last_hidden_state = self.encoder.layer[i].output.LayerNorm(last_hidden_state_before_layernorm)
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
                        # encoder_outputs = self.encoder.layer[i].output.LayerNorm(encoder_outputs)
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
                        last_hidden_state = last_hidden_state_before_layernorm
                        # last_hidden_state = self.encoder.layer[i].output.LayerNorm(last_hidden_state_before_layernorm)
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




            else:
                raise NotImplementedError

            # pooled_output = self.pooler(encoder_outputs[0])
            # res = [output_layers[self.config.num_hidden_layers - 1](pooled_output)]

        return res


        # encoder_outputs = self.encoder(
        #     embedding_output,
        #     extended_attention_mask,
        #     head_mask=head_mask,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )
        #
        # sequence_output = encoder_outputs[0]
        #
        # pooled_output = self.pooler_activation(self.pooler(sequence_output[:, 0])) if self.pooler is not None else None
        #
        # if not return_dict:
        #     return (sequence_output, pooled_output) + encoder_outputs[1:]
        #
        # return BaseModelOutputWithPooling(
        #     last_hidden_state=sequence_output,
        #     pooler_output=pooled_output,
        #     hidden_states=encoder_outputs.hidden_states,
        #     attentions=encoder_outputs.attentions,
        # )


class Albert_SeqLabel_EE(nn.Module):
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

        self.args.keep_norm_same = 0
        # self.ptm_encoder_name = args.ptm_encoder_name
        self.loss_func = nn.CrossEntropyLoss(ignore_index=-100)


        # self.ptm_encoder = get_ptm_from_name(args.ptm_name,self.vocabs['words'],args.ptm_pool_method,
        #                                      args.ptm_word_dropout,layers=args.ptm_layers)
        if args.use_fastnlp_bert:
            raise NotImplementedError
            print('use fastnlp bert!')
            self.ptm_encoder = get_ptm_from_name('bert_cn-wwm', self.vocabs['words'], args.ptm_pool_method,
                                                 0.01, layers=[-1])
        else:
            if 'albert' in args.ptm_name:
                # raise NotImplementedError
                self.ptm_config = AlbertConfig.from_pretrained(args.ptm_name)
                self.ptm_encoder = AlbertModel_EE.from_pretrained(args.ptm_name,args=args)
            else:
                raise NotImplementedError
            # elif 'roberta' in args.ptm_name and args.language_ == 'en':
            #     raise NotImplementedError
            # elif 'roberta' in args.ptm_name and args.language_ == 'cn':
            #     self.ptm_config = BertConfig.from_pretrained(args.ptm_name)
            #     self.ptm_encoder = BertModel_EE.from_pretrained(args.ptm_name,args=args,output_hidden_states=True)
            # elif 'bert' in args.ptm_name:
            #     self.ptm_config = BertConfig.from_pretrained(args.ptm_name)
            #     self.ptm_encoder = BertModel_EE.from_pretrained(args.ptm_name,args=args,output_hidden_states=True)
            #     # raise NotImplementedError
            # else:
            #     raise NotImplementedError
            # print('ptm_model:{}'.format(self.ptm_encoder))
            self.ptm_encoder.resize_token_embeddings(new_num_tokens=len(bundle.tokenizer))

        self.ptm_config.keep_norm_same = args.keep_norm_same
        self.ptm_config.train_mode = args.train_mode
        self.ptm_config.test_mode = args.train_mode
        self.layer_exit_pred = []
        for i in range(self.ptm_config.num_hidden_layers):
            self.layer_exit_pred.append([])

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
            raise NotImplementedError
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
            # print(ptm_output.size())
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

        return result