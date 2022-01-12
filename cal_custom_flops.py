#custom_ops={MyModule:count_my_module}

def count_tener(model,x,y):
    model.total_ops += 0

def count_bert_self_att(model,x,y):
    pass
    # print('x:{}'.format(type(x)))
    # print(len(x))
    # print(x)
    # # hidden_states,attention_mask,head_mask,encoder_hidden_states,encoder_attention_mask,output_attentions = x
    # print('x')
    # for i,p in enumerate(x):
    #     if hasattr(p,'size'):
    #         print(i,p.size())
    #     else:
    #         print(i)
    #
    # print(len(y))
    # print(y)
    # print('y')
    # for i,p in enumerate(y):
    #     if hasattr(p,'size'):
    #         print(i,p.size())
    #     else:
    #         print(i)

def count_bert_self_att_called_by_bertlayer_ee(model,x,y):
    # due to pytorch hook setting, only support all positional parameters sent to BertAttention
    flops = 0
    assert len(x) == 6
    q_hidden_state,a,b,k_v_hidden_state,c,d = x
    batch_size, q_seq_len, hidden_size = q_hidden_state.size()
    assert batch_size==1
    k_v_seq_len = k_v_hidden_state.size(1)
    w_flops = 768*768*(q_seq_len+k_v_seq_len)*2
    attn_flops = (768/12)*q_seq_len*k_v_seq_len*2
    flops+=w_flops
    flops+=attn_flops
    model.total_ops+=flops

def count_albert_self_att_called_by_bertlayer_ee(model,x,y):
    flops = 0
    assert len(x) == 6
    q_hidden_state,a,b,k_v_hidden_state,c,d = x
    batch_size, q_seq_len, hidden_size = q_hidden_state.size()
    assert batch_size==1
    k_v_seq_len = k_v_hidden_state.size(1)
    attn_flops = (768/12)*q_seq_len*k_v_seq_len*2

    flops+=attn_flops
    model.total_ops+=flops

