from fastNLP.io.pipe import Conll2003NERPipe,OntoNotesNERPipe
from paths import *
from fastNLP import cache_results
from transformers import AutoTokenizer,AlbertTokenizer,BertTokenizer
from fastNLP import Vocabulary
import tqdm
import copy

yangjie_rich_pretrain_unigram_path = ""
yangjie_rich_pretrain_bigram_path = ""

@cache_results(_cache_fp='tmp_conll',_refresh=False)
def load_conll(fp, encoding_type='bio', pretrained_model_name_or_path=None):
    # assert pretrained_model_name_or_path
    bundle = Conll2003NERPipe(encoding_type=encoding_type).process_from_file(fp)
    # print(bundle.datasets['train'][:5])
    if 'albert' in pretrained_model_name_or_path:
        tokenizer = AlbertTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path)
        tokenizer.add_tokens(['`', '@'])
        odd_words = set()
        word_to_wordpieces = []
        word_pieces_lengths = []
        vocab = bundle.vocabs['words']
        non_space_token = ['.', ',', ';', '%', ']', ')', '?', '!', '"', "'", '/', ':']
        for word, index in vocab:
            if index == vocab.padding_idx:  # pad是个特殊的符号
                word = tokenizer._pad_token
                print('pad:{}'.format(word))
            elif index == vocab.unknown_idx:
                word = tokenizer._unk_token
                print('unk:{}'.format(word))
            # elif vocab.word_count[word] < min_freq:
            #     word = '[UNK]'
            word_pieces = tokenizer.tokenize(word)
            # word_pieces = tokenizer.convert_ids_to_tokens(word_pieces)
            if word_pieces[0] == '▁':
                word_pieces_ = word_pieces[1:]
                if len(word_pieces_) == 1 and len(word_pieces_[0]) == 1 and word_pieces_[0] in non_space_token:
                    word_pieces = word_pieces_
                else:
                    if word_pieces[0][0] != '▁':
                        print('第一个token非空格，但开头不是空格：{}'.format(word_pieces))
                    word_pieces = word_pieces
                    # odd_words.add((word,tuple(word_pieces)))
            else:
                word_pieces = word_pieces

            word_to_wordpieces.append(word_pieces)
            word_pieces_lengths.append(len(word_pieces))



        # print(odd_words)
        ins_num = 0
        for k,v in bundle.datasets.items():
            ins_num+=len(v)
        pbar = tqdm.tqdm(total=ins_num)

        def get_word_pieces_albert_en(ins):
            words = ins['words']
            raw_words = ins['raw_words']
            word_pieces = []
            raw_word_pieces = []
            now_ins_word_piece_length = []
            first_word_pieces_pos = []

            for i,w in enumerate(words):
                rwp = word_to_wordpieces[w]
                wp = tokenizer.convert_tokens_to_ids(rwp)
                # rwp = tokenizer.tokenize(rw)
                word_pieces.extend(wp)
                raw_word_pieces.extend(rwp)
                now_ins_word_piece_length.append(len(wp))

            for i,l in enumerate(now_ins_word_piece_length):
                if i == 0:
                    first_word_pieces_pos.append(0)
                else:
                    first_word_pieces_pos.append(first_word_pieces_pos[-1]+now_ins_word_piece_length[i-1])

            assert len(first_word_pieces_pos) == len(words)

            first_word_pieces_pos_skip_space = []
            for i,pos in enumerate(first_word_pieces_pos):
                if raw_word_pieces[pos] == '▁':
                    first_word_pieces_pos_skip_space.append(first_word_pieces_pos[i]+1)
                else:
                    first_word_pieces_pos_skip_space.append(first_word_pieces_pos[i])


            last_word_pieces_pos = []
            for i,l in enumerate(now_ins_word_piece_length):
                if i == 0:
                    last_word_pieces_pos.append(now_ins_word_piece_length[0]-1)
                else:
                    last_word_pieces_pos.append(last_word_pieces_pos[-1]+now_ins_word_piece_length[i])

            #add cls sep
            raw_word_pieces.append(tokenizer.sep_token)
            raw_word_pieces.insert(0,tokenizer.cls_token)
            word_pieces.append(tokenizer.sep_token_id)
            word_pieces.insert(0,tokenizer.cls_token_id)
            first_word_pieces_pos = list(map(lambda x:x+1,first_word_pieces_pos))
            first_word_pieces_pos_skip_space = list(map(lambda x: x + 1, first_word_pieces_pos_skip_space))
            last_word_pieces_pos = list(map(lambda x:x+1,last_word_pieces_pos))
            pbar.update(1)
            return raw_word_pieces,word_pieces,now_ins_word_piece_length,first_word_pieces_pos,first_word_pieces_pos_skip_space,last_word_pieces_pos

        for k, v in bundle.datasets.items():
            v.apply(get_word_pieces_albert_en, 'tmp')

    elif  'roberta' in pretrained_model_name_or_path:
        raise NotImplementedError
    elif 'bert' in pretrained_model_name_or_path:
        tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path)
        # tokenizer.add_tokens(['`', '@'])
        odd_words = set()
        word_to_wordpieces = []
        word_pieces_lengths = []
        vocab = bundle.vocabs['words']

        for word, index in vocab:
            if index == vocab.padding_idx:  # pad是个特殊的符号
                word = tokenizer._pad_token
                print('pad:{}'.format(word))
            elif index == vocab.unknown_idx:
                word = tokenizer._unk_token
                print('unk:{}'.format(word))
            # elif vocab.word_count[word] < min_freq:
            #     word = '[UNK]'
            word_pieces = tokenizer.wordpiece_tokenizer.tokenize(word)
            # word_pieces = tokenizer.convert_ids_to_tokens(word_pieces)
            # if word_pieces[0] == '▁':
            #     word_pieces_ = word_pieces[1:]
            #     if len(word_pieces_) == 1 and len(word_pieces_[0]) == 1 and word_pieces_[0] in non_space_token:
            #         word_pieces = word_pieces_
            #     else:
            #         if word_pieces[0][0] != '▁':
            #             print('第一个token非空格，但开头不是空格：{}'.format(word_pieces))
            #         word_pieces = word_pieces
            #         # odd_words.add((word,tuple(word_pieces)))
            # else:
            #     word_pieces = word_pieces

            word_to_wordpieces.append(word_pieces)
            word_pieces_lengths.append(len(word_pieces))

        ins_num = 0
        for k, v in bundle.datasets.items():
            ins_num += len(v)
        pbar = tqdm.tqdm(total=ins_num)

        def get_word_pieces_bert_cn(ins):
            words = ins['words']
            raw_words = ins['raw_words']
            word_pieces = []
            raw_word_pieces = []
            now_ins_word_piece_length = []
            first_word_pieces_pos = []

            for i, w in enumerate(words):
                rwp = word_to_wordpieces[w]
                wp = tokenizer.convert_tokens_to_ids(rwp)
                # rwp = tokenizer.tokenize(rw)
                word_pieces.extend(wp)
                raw_word_pieces.extend(rwp)
                now_ins_word_piece_length.append(len(wp))

            for i, l in enumerate(now_ins_word_piece_length):
                if i == 0:
                    first_word_pieces_pos.append(0)
                else:
                    first_word_pieces_pos.append(first_word_pieces_pos[-1] + now_ins_word_piece_length[i - 1])

            assert len(first_word_pieces_pos) == len(words)

            first_word_pieces_pos_skip_space = copy.deepcopy(first_word_pieces_pos)
            # for i,pos in enumerate(first_word_pieces_pos):
            #     if raw_word_pieces[pos] == '▁':
            #         first_word_pieces_pos_skip_space.append(first_word_pieces_pos[i]+1)
            #     else:
            #         first_word_pieces_pos_skip_space.append(first_word_pieces_pos[i])

            last_word_pieces_pos = []
            for i, l in enumerate(now_ins_word_piece_length):
                if i == 0:
                    last_word_pieces_pos.append(now_ins_word_piece_length[0] - 1)
                else:
                    last_word_pieces_pos.append(last_word_pieces_pos[-1] + now_ins_word_piece_length[i])

            # add cls sep
            raw_word_pieces.append(tokenizer.sep_token)
            raw_word_pieces.insert(0, tokenizer.cls_token)
            word_pieces.append(tokenizer.sep_token_id)
            word_pieces.insert(0, tokenizer.cls_token_id)
            first_word_pieces_pos = list(map(lambda x: x + 1, first_word_pieces_pos))
            first_word_pieces_pos_skip_space = list(map(lambda x: x + 1, first_word_pieces_pos_skip_space))
            last_word_pieces_pos = list(map(lambda x: x + 1, last_word_pieces_pos))
            pbar.update(1)
            return raw_word_pieces, word_pieces, now_ins_word_piece_length, first_word_pieces_pos, first_word_pieces_pos_skip_space, last_word_pieces_pos
        for k, v in bundle.datasets.items():
            v.apply(get_word_pieces_bert_cn, 'tmp')


    for k,v in bundle.datasets.items():
        # v.apply(get_word_pieces_albert_en,'tmp')
        v.apply_field(lambda x:x[0],'tmp','raw_word_pieces')
        v.apply_field(lambda x: x[1], 'tmp', 'word_pieces')
        v.apply_field(lambda x: x[2], 'tmp', 'word_piece_num') # 每个位置的词被拆解为了多少个word piece
        v.apply_field(lambda x: x[3], 'tmp', 'first_word_pieces_pos') # 每个位置的词的第一个word piece在 word piece 序列中的位置
        v.apply_field(lambda x: x[4], 'tmp', 'first_word_pieces_pos_skip_space') # 每个位置的词的第一个word piece在 word piece 序列中的位置，如果有空格，就加一
        v.apply_field(lambda x: x[5], 'tmp', 'last_word_pieces_pos') # 每个位置的词的最后一个word piece在 word piece 序列中的位置，如果有空格，就加一
        v.apply_field(len,'word_pieces','word_piece_seq_len')
        v.apply_field(lambda x:[1]*x,'word_piece_seq_len','bert_attention_mask')


    embeddings = {}
    from fastNLP.embeddings import StaticEmbedding
    embeddings['word_100'] = StaticEmbedding(bundle.vocabs['words'],model_dir_or_name='en-glove-6b-100d',word_dropout=0.01,normalize=True)
    embeddings['word_300'] = StaticEmbedding(bundle.vocabs['words'],model_dir_or_name='en-glove-840b-300d',word_dropout=0.01,normalize=True)
    bundle.tokenizer = tokenizer
    bundle.embeddings = embeddings
    # print(bundle.datasets['dev'])
    return bundle



@cache_results(_cache_fp='tmp_ontonotes_cn',_refresh=False)
def load_ontonotes_cn(fp, encoding_type='bio', pretrained_model_name_or_path=None,dataset_name=''):
    from load_data import load_ontonotes4ner,load_ctb_pos,load_weibo_ner,load_ecom_ner,load_msra_ner,load_ud_pos,load_ud_seg,load_ctb_seg,load_clue_ner
    assert dataset_name != ''
    cache_name = 'cache/{}_{}_1'.format(dataset_name,encoding_type)
    if dataset_name == 'ontonotes_cn':
        bundle = load_ontonotes4ner(fp,index_token=True,char_min_freq=1,bigram_min_freq=1,norm_embed=True,
                                    char_embedding_path=yangjie_rich_pretrain_unigram_path,bigram_embedding_path=yangjie_rich_pretrain_bigram_path,
                                    _cache_fp=cache_name,_refresh=False,encoding_type=encoding_type)
    elif ('ctb' in dataset_name and 'pos' in dataset_name):
        bundle = load_ctb_pos(fp,index_token=True,char_min_freq=1,bigram_min_freq=1,norm_embed=True,
                                    char_embedding_path=yangjie_rich_pretrain_unigram_path,bigram_embedding_path=yangjie_rich_pretrain_bigram_path,
                                    _cache_fp=cache_name,_refresh=False,encoding_type=encoding_type)
    elif ('ctb' in dataset_name and 'seg' in dataset_name):
        bundle = load_ctb_seg(fp,index_token=True,char_min_freq=1,bigram_min_freq=1,norm_embed=True,
                                    char_embedding_path=yangjie_rich_pretrain_unigram_path,bigram_embedding_path=yangjie_rich_pretrain_bigram_path,
                                    _cache_fp=cache_name,_refresh=False,encoding_type=encoding_type)
    elif 'weibo' == dataset_name:
        bundle = load_weibo_ner(fp,unigram_embedding_path=yangjie_rich_pretrain_unigram_path,bigram_embedding_path=yangjie_rich_pretrain_bigram_path,
                                index_token=True,char_min_freq=1,bigram_min_freq=1,encoding_type=encoding_type,norm_embed=True,_cache_fp=cache_name)
    elif 'e_com' == dataset_name:
        bundle = load_ecom_ner(fp,unigram_embedding_path=yangjie_rich_pretrain_unigram_path,bigram_embedding_path=yangjie_rich_pretrain_bigram_path,
                               index_token=True,char_min_freq=1,bigram_min_freq=1,encoding_type=encoding_type,norm_embed=True,_cache_fp=cache_name)
    elif 'msra_ner' == dataset_name:
        bundle = load_msra_ner(fp,char_embedding_path=yangjie_rich_pretrain_unigram_path,bigram_embedding_path=yangjie_rich_pretrain_bigram_path,
                               index_token=True,char_min_freq=1,bigram_min_freq=1,encoding_type=encoding_type,norm_embed=True,_cache_fp=cache_name)
    elif dataset_name[:2] == 'ud' and 'pos' in dataset_name:
        bundle = load_ud_pos(fp,char_embedding_path=yangjie_rich_pretrain_unigram_path,bigram_embedding_path=yangjie_rich_pretrain_bigram_path,
                               index_token=True,char_min_freq=1,bigram_min_freq=1,encoding_type=encoding_type,norm_embed=True,_cache_fp=cache_name)
    elif dataset_name[:2] == 'ud' and 'seg' in dataset_name:
        bundle = load_ud_seg(fp,char_embedding_path=yangjie_rich_pretrain_unigram_path,bigram_embedding_path=yangjie_rich_pretrain_bigram_path,
                               index_token=True,char_min_freq=1,bigram_min_freq=1,encoding_type=encoding_type,norm_embed=True,_cache_fp=cache_name)
    elif dataset_name == 'clue_ner':
        bundle = load_clue_ner(fp,unigram_embedding_path=yangjie_rich_pretrain_unigram_path,bigram_embedding_path=yangjie_rich_pretrain_bigram_path,
                                index_token=True,char_min_freq=1,bigram_min_freq=1,encoding_type=encoding_type,norm_embed=True,_cache_fp=cache_name,
                               _refresh=True)

    else:
        raise NotImplementedError
    print('tokenizer_ptm_name:pretrained_model_name_or_path')
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path)

    # tokenizer.add_tokens(['`', '@'])
    # odd_words = set()
    word_to_wordpieces = []
    word_pieces_lengths = []
    vocab = bundle.vocabs['words']
    non_space_token = ['.', ',', ';', '%', ']', ')', '?', '!', '"', "'", '/', ':']
    for word, index in vocab:
        if index == vocab.padding_idx:  # pad是个特殊的符号
            word = tokenizer._pad_token
            print('pad:{}'.format(word))
        elif index == vocab.unknown_idx:
            word = tokenizer._unk_token
            print('unk:{}'.format(word))
        # elif vocab.word_count[word] < min_freq:
        #     word = '[UNK]'
        word_pieces = tokenizer.wordpiece_tokenizer.tokenize(word)
        # word_pieces = tokenizer.convert_ids_to_tokens(word_pieces)

        # if word_pieces[0] == '▁':
        #     word_pieces_ = word_pieces[1:]
        #     if len(word_pieces_) == 1 and len(word_pieces_[0]) == 1 and word_pieces_[0] in non_space_token:
        #         word_pieces = word_pieces_
        #     else:
        #         if word_pieces[0][0] != '▁':
        #             print('第一个token非空格，但开头不是空格：{}'.format(word_pieces))
        #         word_pieces = word_pieces
        #         # odd_words.add((word,tuple(word_pieces)))
        # else:
        #     word_pieces = word_pieces

        word_to_wordpieces.append(word_pieces)
        word_pieces_lengths.append(len(word_pieces))

    ins_num = 0
    for k,v in bundle.datasets.items():
        ins_num+=len(v)
    pbar = tqdm.tqdm(total=ins_num)

    def get_word_pieces_bert_cn(ins):
        words = ins['words']
        raw_words = ins['raw_words']
        word_pieces = []
        raw_word_pieces = []
        now_ins_word_piece_length = []
        first_word_pieces_pos = []

        for i,w in enumerate(words):
            rwp = word_to_wordpieces[w]
            wp = tokenizer.convert_tokens_to_ids(rwp)
            # rwp = tokenizer.tokenize(rw)
            word_pieces.extend(wp)
            raw_word_pieces.extend(rwp)
            now_ins_word_piece_length.append(len(wp))

        for i,l in enumerate(now_ins_word_piece_length):
            if i == 0:
                first_word_pieces_pos.append(0)
            else:
                first_word_pieces_pos.append(first_word_pieces_pos[-1]+now_ins_word_piece_length[i-1])

        assert len(first_word_pieces_pos) == len(words)

        first_word_pieces_pos_skip_space = copy.deepcopy(first_word_pieces_pos)
        # for i,pos in enumerate(first_word_pieces_pos):
        #     if raw_word_pieces[pos] == '▁':
        #         first_word_pieces_pos_skip_space.append(first_word_pieces_pos[i]+1)
        #     else:
        #         first_word_pieces_pos_skip_space.append(first_word_pieces_pos[i])


        last_word_pieces_pos = []
        for i,l in enumerate(now_ins_word_piece_length):
            if i == 0:
                last_word_pieces_pos.append(now_ins_word_piece_length[0]-1)
            else:
                last_word_pieces_pos.append(last_word_pieces_pos[-1]+now_ins_word_piece_length[i])

        #add cls sep
        raw_word_pieces.append(tokenizer.sep_token)
        raw_word_pieces.insert(0,tokenizer.cls_token)
        word_pieces.append(tokenizer.sep_token_id)
        word_pieces.insert(0,tokenizer.cls_token_id)
        first_word_pieces_pos = list(map(lambda x:x+1,first_word_pieces_pos))
        first_word_pieces_pos_skip_space = list(map(lambda x: x + 1, first_word_pieces_pos_skip_space))
        last_word_pieces_pos = list(map(lambda x:x+1,last_word_pieces_pos))
        pbar.update(1)
        return raw_word_pieces,word_pieces,now_ins_word_piece_length,first_word_pieces_pos,first_word_pieces_pos_skip_space,last_word_pieces_pos

    for k,v in bundle.datasets.items():
        v.apply(get_word_pieces_bert_cn,'tmp')
        v.apply_field(lambda x:x[0],'tmp','raw_word_pieces')
        v.apply_field(lambda x: x[1], 'tmp', 'word_pieces')
        v.apply_field(lambda x: x[2], 'tmp', 'word_piece_num') # 每个位置的词被拆解为了多少个word piece
        v.apply_field(lambda x: x[3], 'tmp', 'first_word_pieces_pos') # 每个位置的词的第一个word piece在 word piece 序列中的位置
        v.apply_field(lambda x: x[4], 'tmp', 'first_word_pieces_pos_skip_space') # 每个位置的词的第一个word piece在 word piece 序列中的位置，如果有空格，就加一
        v.apply_field(lambda x: x[5], 'tmp', 'last_word_pieces_pos') # 每个位置的词的最后一个word piece在 word piece 序列中的位置，如果有空格，就加一
        v.apply_field(len,'word_pieces','word_piece_seq_len')
        v.apply_field(lambda x:[1]*x,'word_piece_seq_len','bert_attention_mask')


    bundle.tokenizer = tokenizer

    return bundle


@cache_results(_cache_fp='tmp_ontonotes',_refresh=True)
def load_ontonotes(fp,encoding_type='bio'):
    raise NotImplementedError
    bundle = OntoNotesNERPipe(encoding_type=encoding_type).process_from_file(fp)
    return bundle


@cache_results(_cache_fp='tmp_conll',_refresh=False)
def load_conll_two_col(fp, encoding_type='bio', pretrained_model_name_or_path=None):
    assert encoding_type == 'bio'
    # assert pretrained_model_name_or_path
    from fastNLP.io import Conll2003Loader
    from tmp_fastnlp_module import MyConllLoader
    from fastNLP import Vocabulary
    from fastNLP.io.data_bundle import DataBundle
    loader = MyConllLoader(headers=['raw_words','raw_target'],sep='\t',dropna=False)
    # bundle = Conll2003NERPipe(encoding_type=encoding_type).process_from_file(fp)
    train_file_name = 'train.conll'
    dev_file_name = 'dev.conll'
    test_file_name = 'test.conll'
    fp_dict = {'train': train_file_name, 'dev': dev_file_name, 'test': test_file_name}

    datasets = {}
    for k,v in fp_dict.items():
        tmp_dataset = loader.load('{}/{}'.format(fp,v)).datasets['train']
        datasets[k] = tmp_dataset
    # print(datasets['train'])

    # for ins in datasets['train'][:20]:
    #     raw_words = ins['raw_words']
    #     raw_target = ins['raw_target']
    #     print('{}/{}'.format(raw_words,raw_target))

    word_vocab = Vocabulary()
    target_vocab = Vocabulary(padding=None,unknown=None)
    word_vocab.from_dataset(datasets['train'],field_name='raw_words',no_create_entry_dataset=[datasets['dev'],datasets['test']])
    target_vocab.from_dataset(datasets['train'],field_name='raw_target')

    print(target_vocab.word2idx)

    word_vocab.index_dataset(datasets['train'], datasets['dev'], datasets['test'],
                             field_name='raw_words', new_field_name='words')
    target_vocab.index_dataset(datasets['train'], datasets['dev'], datasets['test'],
                               field_name='raw_target',new_field_name='target')
    bundle = DataBundle()
    bundle.datasets = datasets
    bundle.vocabs = {'words':word_vocab,'target':target_vocab}
    bundle.apply_field(lambda x:len(x),field_name='words',new_field_name='seq_len')

    # print(datasets['train'][:5])
    # exit()
    if 'albert' in pretrained_model_name_or_path:
        tokenizer = AlbertTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path)
        tokenizer.add_tokens(['`', '@'])
        odd_words = set()
        word_to_wordpieces = []
        word_pieces_lengths = []
        vocab = bundle.vocabs['words']
        non_space_token = ['.', ',', ';', '%', ']', ')', '?', '!', '"', "'", '/', ':']
        for word, index in vocab:
            if index == vocab.padding_idx:  # pad是个特殊的符号
                word = tokenizer._pad_token
                print('pad:{}'.format(word))
            elif index == vocab.unknown_idx:
                word = tokenizer._unk_token
                print('unk:{}'.format(word))
            # elif vocab.word_count[word] < min_freq:
            #     word = '[UNK]'
            word_pieces = tokenizer.tokenize(word)
            # word_pieces = tokenizer.convert_ids_to_tokens(word_pieces)
            if word_pieces[0] == '▁':
                word_pieces_ = word_pieces[1:]
                if len(word_pieces_) == 1 and len(word_pieces_[0]) == 1 and word_pieces_[0] in non_space_token:
                    word_pieces = word_pieces_
                else:
                    if word_pieces[0][0] != '▁':
                        print('第一个token非空格，但开头不是空格：{}'.format(word_pieces))
                    word_pieces = word_pieces
                    # odd_words.add((word,tuple(word_pieces)))
            else:
                word_pieces = word_pieces

            word_to_wordpieces.append(word_pieces)
            word_pieces_lengths.append(len(word_pieces))



        # print(odd_words)
        ins_num = 0
        for k,v in bundle.datasets.items():
            ins_num+=len(v)
        pbar = tqdm.tqdm(total=ins_num)

        def get_word_pieces_albert_en(ins):
            words = ins['words']
            raw_words = ins['raw_words']
            word_pieces = []
            raw_word_pieces = []
            now_ins_word_piece_length = []
            first_word_pieces_pos = []

            for i,w in enumerate(words):
                rwp = word_to_wordpieces[w]
                wp = tokenizer.convert_tokens_to_ids(rwp)
                # rwp = tokenizer.tokenize(rw)
                word_pieces.extend(wp)
                raw_word_pieces.extend(rwp)
                now_ins_word_piece_length.append(len(wp))

            for i,l in enumerate(now_ins_word_piece_length):
                if i == 0:
                    first_word_pieces_pos.append(0)
                else:
                    first_word_pieces_pos.append(first_word_pieces_pos[-1]+now_ins_word_piece_length[i-1])

            assert len(first_word_pieces_pos) == len(words)

            first_word_pieces_pos_skip_space = []
            for i,pos in enumerate(first_word_pieces_pos):
                if raw_word_pieces[pos] == '▁':
                    first_word_pieces_pos_skip_space.append(first_word_pieces_pos[i]+1)
                else:
                    first_word_pieces_pos_skip_space.append(first_word_pieces_pos[i])


            last_word_pieces_pos = []
            for i,l in enumerate(now_ins_word_piece_length):
                if i == 0:
                    last_word_pieces_pos.append(now_ins_word_piece_length[0]-1)
                else:
                    last_word_pieces_pos.append(last_word_pieces_pos[-1]+now_ins_word_piece_length[i])

            #add cls sep
            raw_word_pieces.append(tokenizer.sep_token)
            raw_word_pieces.insert(0,tokenizer.cls_token)
            word_pieces.append(tokenizer.sep_token_id)
            word_pieces.insert(0,tokenizer.cls_token_id)
            first_word_pieces_pos = list(map(lambda x:x+1,first_word_pieces_pos))
            first_word_pieces_pos_skip_space = list(map(lambda x: x + 1, first_word_pieces_pos_skip_space))
            last_word_pieces_pos = list(map(lambda x:x+1,last_word_pieces_pos))
            pbar.update(1)
            return raw_word_pieces,word_pieces,now_ins_word_piece_length,first_word_pieces_pos,first_word_pieces_pos_skip_space,last_word_pieces_pos

        for k, v in bundle.datasets.items():
            v.apply(get_word_pieces_albert_en, 'tmp')

    elif  'roberta' in pretrained_model_name_or_path:
        raise NotImplementedError
    elif 'bert' in pretrained_model_name_or_path:
        tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path)
        # tokenizer.add_tokens(['`', '@'])
        odd_words = set()
        word_to_wordpieces = []
        word_pieces_lengths = []
        vocab = bundle.vocabs['words']

        for word, index in vocab:
            if index == vocab.padding_idx:  # pad是个特殊的符号
                word = tokenizer._pad_token
                print('pad:{}'.format(word))
            elif index == vocab.unknown_idx:
                word = tokenizer._unk_token
                print('unk:{}'.format(word))
            # elif vocab.word_count[word] < min_freq:
            #     word = '[UNK]'
            word_pieces = tokenizer.wordpiece_tokenizer.tokenize(word)
            # word_pieces = tokenizer.convert_ids_to_tokens(word_pieces)
            # if word_pieces[0] == '▁':
            #     word_pieces_ = word_pieces[1:]
            #     if len(word_pieces_) == 1 and len(word_pieces_[0]) == 1 and word_pieces_[0] in non_space_token:
            #         word_pieces = word_pieces_
            #     else:
            #         if word_pieces[0][0] != '▁':
            #             print('第一个token非空格，但开头不是空格：{}'.format(word_pieces))
            #         word_pieces = word_pieces
            #         # odd_words.add((word,tuple(word_pieces)))
            # else:
            #     word_pieces = word_pieces

            word_to_wordpieces.append(word_pieces)
            word_pieces_lengths.append(len(word_pieces))

        ins_num = 0
        for k, v in bundle.datasets.items():
            ins_num += len(v)
        pbar = tqdm.tqdm(total=ins_num)

        def get_word_pieces_bert_cn(ins):
            words = ins['words']
            raw_words = ins['raw_words']
            word_pieces = []
            raw_word_pieces = []
            now_ins_word_piece_length = []
            first_word_pieces_pos = []

            for i, w in enumerate(words):
                rwp = word_to_wordpieces[w]
                wp = tokenizer.convert_tokens_to_ids(rwp)
                # rwp = tokenizer.tokenize(rw)
                word_pieces.extend(wp)
                raw_word_pieces.extend(rwp)
                now_ins_word_piece_length.append(len(wp))

            for i, l in enumerate(now_ins_word_piece_length):
                if i == 0:
                    first_word_pieces_pos.append(0)
                else:
                    first_word_pieces_pos.append(first_word_pieces_pos[-1] + now_ins_word_piece_length[i - 1])

            assert len(first_word_pieces_pos) == len(words)

            first_word_pieces_pos_skip_space = copy.deepcopy(first_word_pieces_pos)
            # for i,pos in enumerate(first_word_pieces_pos):
            #     if raw_word_pieces[pos] == '▁':
            #         first_word_pieces_pos_skip_space.append(first_word_pieces_pos[i]+1)
            #     else:
            #         first_word_pieces_pos_skip_space.append(first_word_pieces_pos[i])

            last_word_pieces_pos = []
            for i, l in enumerate(now_ins_word_piece_length):
                if i == 0:
                    last_word_pieces_pos.append(now_ins_word_piece_length[0] - 1)
                else:
                    last_word_pieces_pos.append(last_word_pieces_pos[-1] + now_ins_word_piece_length[i])

            # add cls sep
            raw_word_pieces.append(tokenizer.sep_token)
            raw_word_pieces.insert(0, tokenizer.cls_token)
            word_pieces.append(tokenizer.sep_token_id)
            word_pieces.insert(0, tokenizer.cls_token_id)
            first_word_pieces_pos = list(map(lambda x: x + 1, first_word_pieces_pos))
            first_word_pieces_pos_skip_space = list(map(lambda x: x + 1, first_word_pieces_pos_skip_space))
            last_word_pieces_pos = list(map(lambda x: x + 1, last_word_pieces_pos))
            pbar.update(1)
            return raw_word_pieces, word_pieces, now_ins_word_piece_length, first_word_pieces_pos, first_word_pieces_pos_skip_space, last_word_pieces_pos
        for k, v in bundle.datasets.items():
            v.apply(get_word_pieces_bert_cn, 'tmp')


    for k,v in bundle.datasets.items():
        # v.apply(get_word_pieces_albert_en,'tmp')
        v.apply_field(lambda x:x[0],'tmp','raw_word_pieces')
        v.apply_field(lambda x: x[1], 'tmp', 'word_pieces')
        v.apply_field(lambda x: x[2], 'tmp', 'word_piece_num') # 每个位置的词被拆解为了多少个word piece
        v.apply_field(lambda x: x[3], 'tmp', 'first_word_pieces_pos') # 每个位置的词的第一个word piece在 word piece 序列中的位置
        v.apply_field(lambda x: x[4], 'tmp', 'first_word_pieces_pos_skip_space') # 每个位置的词的第一个word piece在 word piece 序列中的位置，如果有空格，就加一
        v.apply_field(lambda x: x[5], 'tmp', 'last_word_pieces_pos') # 每个位置的词的最后一个word piece在 word piece 序列中的位置，如果有空格，就加一
        v.apply_field(len,'word_pieces','word_piece_seq_len')
        v.apply_field(lambda x:[1]*x,'word_piece_seq_len','bert_attention_mask')


    embeddings = {}
    from fastNLP.embeddings import StaticEmbedding
    embeddings['word_100'] = StaticEmbedding(bundle.vocabs['words'],model_dir_or_name='en-glove-6b-100d',word_dropout=0.01,normalize=True)
    embeddings['word_300'] = StaticEmbedding(bundle.vocabs['words'],model_dir_or_name='en-glove-840b-300d',word_dropout=0.01,normalize=True)
    bundle.tokenizer = tokenizer
    bundle.embeddings = embeddings
    # print(bundle.datasets['dev'])
    return bundle


if __name__ == '__main__':
    from paths import ritter_path,ark_twitter_path,fjl_twitter_ner_path
    fp = ark_twitter_path
    bundle = load_conll_two_col(fp,'bio',pretrained_model_name_or_path='bert-base-cased',
                                _cache_fp='cache/tmp_{}'.format('_'.join(fp.split('/'))),
                                _refresh=False)

    transitions = set()

    for ins in bundle.datasets['train']:
        raw_target = ins['raw_target']
        for i in range(len(raw_target)-1):
            transitions.add((raw_target[i],raw_target[i+1]))

    for ins in bundle.datasets['dev']:
        raw_target = ins['raw_target']
        for i in range(len(raw_target) - 1):
            if (raw_target[i],raw_target[i+1]) not in transitions:
                print((raw_target[i],raw_target[i+1]))
    print('*'*20)
    for ins in bundle.datasets['test']:
        raw_target = ins['raw_target']
        for i in range(len(raw_target) - 1):
            if (raw_target[i],raw_target[i+1]) not in transitions:
                print((raw_target[i],raw_target[i+1]))



    exit()





    bundle = load_ontonotes_cn(ontonote4ner_cn_path,pretrained_model_name_or_path='hfl/chinese-bert-wwm',_refresh=False)
    print(bundle)

    train_data = bundle.datasets['train']
    ins_indexs = [500,76]

    bundle = load_conll(conll_path, 'bio', pretrained_model_name_or_path='bert-base-cased',
                         _refresh=False,_cache_fp='tmp_conll_202113')

    train_data = bundle.datasets['train']

    for i in range(10,20):
        ins = train_data[i]
        raw_words = ins['raw_words']
        raw_word_pieces = bundle.tokenizer.convert_ids_to_tokens(ins['word_pieces'])

        print('{}:rw:{}'.format(i,raw_words))
        print('{}:rwp:{}'.format(i,raw_word_pieces))
    exit()

    # for ins_index in ins_indexs:

    for k,v in bundle.datasets.items():
        for i,ins in enumerate(v):
            seq_len = ins['seq_len']
            word_piece_seq_len = ins['word_piece_seq_len']
            # assert seq_len == word_piece_seq_len
            if seq_len!=(word_piece_seq_len-2):
                raw_word_pieces = ins['raw_word_pieces']
                raw_words = ins['raw_words']
                print('-'*40)
                print('for show, raw_words:{}'.format(raw_words))
                print('for show, first_raw_word_piece:{}'.format(raw_word_pieces))

                print('-'*40)

            if i in ins_indexs:
                raw_word_pieces = ins['raw_word_pieces']
                raw_words = ins['raw_words']
                print('-'*40)
                print('for show, raw_words:{}'.format(raw_words))
                print('for show, first_raw_word_piece:{}'.format(raw_word_pieces))

                raw_word_pieces = ins['raw_word_pieces']
                raw_words = ins['raw_words']

                first_word_pieces_pos_skip_space = ins['first_word_pieces_pos_skip_space']
                first_word_pieces_pos = ins['first_word_pieces_pos']
                last_word_pieces_pos = ins['last_word_pieces_pos']

                print('first_word_pieces_pos:{}'.format(first_word_pieces_pos))
                print('first_word_pieces_pos_skip_space:{}'.format(first_word_pieces_pos_skip_space))
                print('last_word_pieces_pos:{}'.format(last_word_pieces_pos))


                print('-'*40)



    exit()


    print('unk:{}'.format(bundle.tokenizer.unk_token_id))
    for k,v in bundle.datasets.items():
        unk_token = set()
        for ins_index,ins in enumerate(v):
            first_word_pieces_pos_skip_space = ins['first_word_pieces_pos_skip_space']
            last_word_pieces_pos = ins['last_word_pieces_pos']
            raw_word_pieces = ins['raw_word_pieces']
            raw_words = ins['raw_words']

            if ins_index in ins_indexs:
                first_rwp_list = []
                for i, index in enumerate(first_word_pieces_pos_skip_space):
                    rw = raw_words[i]
                    first_rwp = raw_word_pieces[first_word_pieces_pos_skip_space[i]]
                    first_rwp_list.append(first_rwp)
                print('-'*40)
                print('for show, raw_words:{}'.format(raw_words))
                print('for show, first_raw_word_piece:{}'.format(first_rwp_list))

                print('-'*40)


            for i, index in enumerate(first_word_pieces_pos_skip_space):
                rw = raw_words[i].lower()
                first_rwp = raw_word_pieces[first_word_pieces_pos_skip_space[i]].lower()
                if first_rwp[0] == '▁':
                    first_rwp = first_rwp[1:]
                if rw[:len(first_rwp)] != first_rwp:
                    print('*'*40)
                    print('wrong at')
                    print('raw_words:{}'.format(raw_words))
                    print('raw_word_pieces:{}'.format(raw_word_pieces))
                    print('first_word_pieces_pos_skip_space:{}'.format(first_word_pieces_pos_skip_space))
                    print('*'*40)
                    continue



            # ins = train_data[ins_index]
            # for k,v in ins.items():
            #     print('{}:{}'.format(k,v))
            # print('***************'*8)

            # word_pieces = ins['word_pieces']
            # if '@' in ins['raw_word_pieces'] or '`' in ins['raw_word_pieces']:
            #     print(ins['raw_word_pieces'])
            #     print(word_pieces)
            #     print('*'*50)
                # exit()
            # assert bundle.tokenizer.unk_token_id not in word_pieces
            # if bundle.tokenizer.unk_token_id in word_pieces:
            #     for i,wp in enumerate(word_pieces):
            #         if wp == bundle.tokenizer.unk_token_id:
            #             unk_token.add(ins['raw_word_pieces'][i])
                # print(ins['raw_word_pieces'])
                # print(ins['word_pieces'])
                # print(ins['raw_words'])
                # print('*'*40)
        # print('{}:{}'.format(k,unk_token))

        # words = ins['words']
        # raw_words = ins['raw_words']

    # for k,v in ins:
    #     print('{}:{}'.format(k,v))
