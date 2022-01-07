from fastNLP.io.pipe import Conll2003NERPipe,OntoNotesNERPipe
from paths import *
from fastNLP import cache_results
from transformers import AutoTokenizer
import os

@cache_results(_cache_fp='tmp_conll')
def load_conll(fp,encoding_type='bio',pretrained_model_name_or_path=None):
    # assert pretrained_model_name_or_path
    bundle = Conll2003NERPipe(encoding_type=encoding_type).process_from_file(fp)
    # tokenizer = AutoTokenizer(pretrained_model_name_or_path=pretrained_model_name_or_path)

    # print(bundle.datasets['dev'])
    return bundle

@cache_results(_cache_fp='tmp_ontonotes',_refresh=True)
def load_ontonotes(fp,encoding_type='bio'):
    bundle = OntoNotesNERPipe(encoding_type=encoding_type).process_from_file(fp)
    return bundle


@cache_results(_cache_fp='cache/ontonotes4ner',_refresh=False)
def load_ontonotes4ner(path,char_embedding_path=None,bigram_embedding_path=None,index_token=True,
                       char_min_freq=1,bigram_min_freq=1,only_train_min_freq=0,norm_embed=False,encoding_type='bmeso'):
    assert encoding_type in ['bmeso','bioes','bio']
    from fastNLP.io.loader import ConllLoader
    from utils import get_bigrams
    import os
    from fastNLP import Vocabulary
    from fastNLP.embeddings import StaticEmbedding
    from fastNLP.io.data_bundle import DataBundle

    train_path = os.path.join(path,'train.char.bmes{}'.format(''))
    dev_path = os.path.join(path,'dev.char.bmes')
    test_path = os.path.join(path,'test.char.bmes')

    loader = ConllLoader(['chars','target'])
    train_bundle = loader.load(train_path)
    dev_bundle = loader.load(dev_path)
    test_bundle = loader.load(test_path)


    datasets = dict()
    datasets['train'] = train_bundle.datasets['train']
    datasets['dev'] = dev_bundle.datasets['train']
    datasets['test'] = test_bundle.datasets['train']


    datasets['train'].apply_field(get_bigrams,field_name='chars',new_field_name='bigrams')
    datasets['dev'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')
    datasets['test'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')

    datasets['train'].add_seq_len('chars')
    datasets['dev'].add_seq_len('chars')
    datasets['test'].add_seq_len('chars')



    char_vocab = Vocabulary()
    bigram_vocab = Vocabulary()
    label_vocab = Vocabulary(padding=None,unknown=None)
    # print(datasets.keys())
    # print(len(datasets['dev']))
    # print(len(datasets['test']))
    # print(len(datasets['train']))
    char_vocab.from_dataset(datasets['train'],field_name='chars',
                            no_create_entry_dataset=[datasets['dev'],datasets['test']])
    bigram_vocab.from_dataset(datasets['train'],field_name='bigrams',
                              no_create_entry_dataset=[datasets['dev'],datasets['test']])
    if 'ctb9' in path:
        label_vocab.from_dataset(datasets['train'],datasets['dev'],datasets['test'],field_name='target')
    else:
        label_vocab.from_dataset(datasets['train'],field_name='target')
    if index_token:
        for k,v in datasets.items():
            v.rename_field('chars','raw_chars')
            v.rename_field('bigrams','raw_bigrams')
        char_vocab.index_dataset(datasets['train'],datasets['dev'],datasets['test'],
                                 field_name='raw_chars',new_field_name='chars')
        bigram_vocab.index_dataset(datasets['train'],datasets['dev'],datasets['test'],
                                 field_name='raw_bigrams',new_field_name='bigrams')
        label_vocab.index_dataset(datasets['train'],datasets['dev'],datasets['test'],
                                 field_name='target',new_field_name='target')

    vocabs = {}
    vocabs['words'] = char_vocab
    # vocabs['label'] = label_vocab
    vocabs['bigrams'] = bigram_vocab
    vocabs['target'] = label_vocab

    embeddings = {}
    if char_embedding_path is not None:
        char_embedding = StaticEmbedding(char_vocab,char_embedding_path,word_dropout=0.01,
                                         min_freq=char_min_freq,only_train_min_freq=only_train_min_freq,normalize=norm_embed)
        embeddings['char'] = char_embedding

    if bigram_embedding_path is not None:
        bigram_embedding = StaticEmbedding(bigram_vocab,bigram_embedding_path,word_dropout=0.01,
                                           min_freq=bigram_min_freq,only_train_min_freq=only_train_min_freq,normalize=norm_embed)
        embeddings['bigram'] = bigram_embedding

    for k,v in datasets.items():
        v.rename_field('chars','words')
        v.rename_field('raw_chars','raw_words')
    bundle = DataBundle()
    bundle.datasets = datasets
    bundle.vocabs = vocabs
    bundle.embeddings = embeddings
    from utils import transform_bmeso_bundle_to_bio
    if encoding_type == 'bmeso':
        return bundle
    elif encoding_type == 'bio':
        bundle = transform_bmeso_bundle_to_bio(bundle)
        return bundle
    elif encoding_type == 'bioes':
        return bundle

@cache_results(_cache_fp='cache/weiboNER_uni+bi', _refresh=True)
def load_weibo_ner(path,unigram_embedding_path=None,bigram_embedding_path=None,index_token=True,
                   char_min_freq=1,bigram_min_freq=1,only_train_min_freq=0,char_word_dropout=0.01,encoding_type='bio',norm_embed=True):
    from fastNLP.io.loader import ConllLoader
    from utils import get_bigrams

    assert encoding_type in ['bmeso','bioes','bio']
    from fastNLP.io.loader import ConllLoader
    from utils import get_bigrams
    import os
    from fastNLP import Vocabulary
    from fastNLP.embeddings import StaticEmbedding
    from fastNLP.io.data_bundle import DataBundle

    loader = ConllLoader(['chars','target'])
    # bundle = loader.load(path)
    #
    # datasets = bundle.datasets

    # print(datasets['train'][:5])

    train_path = os.path.join(path,'weiboNER_2nd_conll.train_deseg')
    dev_path = os.path.join(path, 'weiboNER_2nd_conll.dev_deseg')
    test_path = os.path.join(path, 'weiboNER_2nd_conll.test_deseg')

    paths = {}
    paths['train'] = train_path
    paths['dev'] = dev_path
    paths['test'] = test_path

    datasets = {}

    for k,v in paths.items():
        bundle = loader.load(v)
        datasets[k] = bundle.datasets['train']



    for k,v in datasets.items():
        print('{}:{}'.format(k,len(v)))
    # print(*list(datasets.keys()))
    vocabs = {}
    char_vocab = Vocabulary()
    bigram_vocab = Vocabulary()
    label_vocab = Vocabulary(padding=None,unknown=None)

    for k,v in datasets.items():
        # ignore the word segmentation tag
        v.apply_field(lambda x: [w[0] for w in x],'chars','chars')
        v.apply_field(get_bigrams,'chars','bigrams')


    char_vocab.from_dataset(datasets['train'],field_name='chars',no_create_entry_dataset=[datasets['dev'],datasets['test']])
    label_vocab.from_dataset(datasets['train'],field_name='target')
    print('label_vocab:{}\n{}'.format(len(label_vocab),label_vocab.idx2word))


    for k,v in datasets.items():
        # v.set_pad_val('target',-100)
        v.add_seq_len('chars',new_field_name='seq_len')


    vocabs['words'] = char_vocab
    vocabs['target'] = label_vocab


    bigram_vocab.from_dataset(datasets['train'],field_name='bigrams',no_create_entry_dataset=[datasets['dev'],datasets['test']])
    if index_token:
        for k,v in datasets.items():
            v.rename_field('chars','raw_chars')
            v.rename_field('bigrams','raw_bigrams')
        char_vocab.index_dataset(*list(datasets.values()), field_name='raw_chars', new_field_name='chars')
        bigram_vocab.index_dataset(*list(datasets.values()),field_name='raw_bigrams',new_field_name='bigrams')
        label_vocab.index_dataset(*list(datasets.values()), field_name='target', new_field_name='target')

    # for k,v in datasets.items():
    #     v.set_input('chars','bigrams','seq_len','target')
    #     v.set_target('target','seq_len')

    vocabs['bigram'] = bigram_vocab


    embeddings = {}

    if unigram_embedding_path is not None:
        unigram_embedding = StaticEmbedding(char_vocab, model_dir_or_name=unigram_embedding_path,
                                            word_dropout=char_word_dropout,
                                            min_freq=char_min_freq,only_train_min_freq=only_train_min_freq,normalize=norm_embed)
        embeddings['char'] = unigram_embedding

    if bigram_embedding_path is not None:
        bigram_embedding = StaticEmbedding(bigram_vocab, model_dir_or_name=bigram_embedding_path,
                                           word_dropout=0.01,
                                           min_freq=bigram_min_freq,only_train_min_freq=only_train_min_freq,normalize=norm_embed)
        embeddings['bigram'] = bigram_embedding

    for k,v in datasets.items():
        v.rename_field('chars','words')
        v.rename_field('raw_chars','raw_words')
    bundle = DataBundle()
    bundle.datasets = datasets
    bundle.vocabs = vocabs
    bundle.embeddings = embeddings
    from utils import transform_bmeso_bundle_to_bio,transform_bio_bundle_to_bioes
    if encoding_type == 'bmeso':
        bundle = transform_bio_bundle_to_bioes(bundle)
        return bundle
        pass
    elif encoding_type == 'bio':
        # bundle = transform_bmeso_bundle_to_bio(bundle)
        return bundle
        pass
    elif encoding_type == 'bioes':
        bundle = transform_bio_bundle_to_bioes(bundle)
        return bundle
        pass

    # return datasets, vocabs, embeddings

from fastNLP.io.loader import ConllLoader
from fastNLP import DataSet,Instance
from fastNLP.core import logger
class Ecom_Conll_Loader(ConllLoader):
    def _load(self, path):
        r"""
        传入的一个文件路径，将该文件读入DataSet中，field由ConllLoader初始化时指定的headers决定。

        :param str path: 文件的路径
        :return: DataSet
        """

        def _read_conll(path, encoding='utf-8', sep=None, indexes=None, dropna=True):
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
                    line = line.strip('\n')
                    # line = line[:-1]
                    if line == '':
                        if len(sample):
                            try:
                                res = parse_conll(sample)
                                sample = []
                                yield line_idx, res
                            except Exception as e:
                                if dropna:
                                    logger.warning(
                                        'Invalid instance which ends at line: {} has been dropped.'.format(line_idx))
                                    sample = []
                                    continue
                                raise ValueError('Invalid instance which ends at line: {}'.format(line_idx))
                    elif line.startswith('#'):
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


        ds = DataSet()
        for idx, data in _read_conll(path,sep=self.sep, indexes=self.indexes, dropna=self.dropna):
            ins = {h: data[i] for i, h in enumerate(self.headers)}
            ds.append(Instance(**ins))
        return ds


@cache_results(_cache_fp='cache/ecom_NER_uni+bi', _refresh=True)
def load_ecom_ner(path,unigram_embedding_path=None,bigram_embedding_path=None,index_token=True,
                   char_min_freq=1,bigram_min_freq=1,only_train_min_freq=0,char_word_dropout=0.01,encoding_type='bioes',norm_embed=True):
    from fastNLP.io.loader import ConllLoader
    from utils import get_bigrams

    assert encoding_type in ['bmeso','bioes','bio']
    # from fastNLP.io.loader import ConllLoader,CSVLoader
    from utils import get_bigrams
    import os
    from fastNLP import Vocabulary
    from fastNLP.embeddings import StaticEmbedding
    from fastNLP.io.data_bundle import DataBundle

    loader = Ecom_Conll_Loader(['chars','target'],sep='\t')
    # loader = CSVLoader(['chars','target'],sep='\t')
    # bundle = loader.load(path)
    #
    # datasets = bundle.datasets

    # print(datasets['train'][:5])

    train_path = os.path.join(path,'train.txt.bieos')
    dev_path = os.path.join(path, 'dev.txt.bieos')
    test_path = os.path.join(path, 'test.txt.bieos')

    paths = {}
    paths['train'] = train_path
    paths['dev'] = dev_path
    paths['test'] = test_path

    datasets = {}

    for k,v in paths.items():
        bundle = loader.load(v)
        datasets[k] = bundle.datasets['train']
        # print(datasets[k][:5])




    for k,v in datasets.items():
        print('{}:{}'.format(k,len(v)))
    # print(*list(datasets.keys()))
    vocabs = {}
    char_vocab = Vocabulary()
    bigram_vocab = Vocabulary()
    label_vocab = Vocabulary(padding=None,unknown=None)

    for k,v in datasets.items():
        # ignore the word segmentation tag
        v.apply_field(lambda x: [w[0] for w in x],'chars','chars')
        v.apply_field(get_bigrams,'chars','bigrams')


    char_vocab.from_dataset(datasets['train'],field_name='chars',no_create_entry_dataset=[datasets['dev'],datasets['test']])
    label_vocab.from_dataset(datasets['train'],field_name='target')
    print('label_vocab:{}\n{}'.format(len(label_vocab),label_vocab.idx2word))


    for k,v in datasets.items():
        # v.set_pad_val('target',-100)
        v.add_seq_len('chars',new_field_name='seq_len')


    vocabs['words'] = char_vocab
    vocabs['target'] = label_vocab


    bigram_vocab.from_dataset(datasets['train'],field_name='bigrams',no_create_entry_dataset=[datasets['dev'],datasets['test']])
    if index_token:
        for k,v in datasets.items():
            v.rename_field('chars','raw_chars')
            v.rename_field('bigrams','raw_bigrams')
        char_vocab.index_dataset(*list(datasets.values()), field_name='raw_chars', new_field_name='chars')
        bigram_vocab.index_dataset(*list(datasets.values()),field_name='raw_bigrams',new_field_name='bigrams')
        label_vocab.index_dataset(*list(datasets.values()), field_name='target', new_field_name='target')

    # for k,v in datasets.items():
    #     v.set_input('chars','bigrams','seq_len','target')
    #     v.set_target('target','seq_len')

    vocabs['bigram'] = bigram_vocab


    embeddings = {}

    if unigram_embedding_path is not None:
        unigram_embedding = StaticEmbedding(char_vocab, model_dir_or_name=unigram_embedding_path,
                                            word_dropout=char_word_dropout,
                                            min_freq=char_min_freq,only_train_min_freq=only_train_min_freq,normalize=norm_embed)
        embeddings['char'] = unigram_embedding

    if bigram_embedding_path is not None:
        bigram_embedding = StaticEmbedding(bigram_vocab, model_dir_or_name=bigram_embedding_path,
                                           word_dropout=0.01,
                                           min_freq=bigram_min_freq,only_train_min_freq=only_train_min_freq,normalize=norm_embed)
        embeddings['bigram'] = bigram_embedding

    for k,v in datasets.items():
        v.rename_field('chars','words')
        v.rename_field('raw_chars','raw_words')
    bundle = DataBundle()
    bundle.datasets = datasets
    bundle.vocabs = vocabs
    bundle.embeddings = embeddings
    from utils import transform_bmeso_bundle_to_bio,transform_bio_bundle_to_bioes
    if encoding_type == 'bmeso':
        # bundle = transform_bio_bundle_to_bioes(bundle)

        return bundle
        pass
    elif encoding_type == 'bio':
        bundle = transform_bmeso_bundle_to_bio(bundle)
        return bundle
        pass
    elif encoding_type == 'bioes':
        # bundle = transform_bio_bundle_to_bioes(bundle)
        return bundle
        pass

    return datasets, vocabs, embeddings





@cache_results(_cache_fp='cache/ctb', _refresh=True)
def load_ctb_pos(path,char_embedding_path=None,bigram_embedding_path=None,index_token=True,
                   char_min_freq=1,bigram_min_freq=1,only_train_min_freq=0,char_word_dropout=0.01,encoding_type='bio',norm_embed=True):
    from fastNLP.io.loader import ConllLoader
    from utils import get_bigrams

    assert encoding_type in ['bmeso','bioes','bio']
    from fastNLP.io.loader import ConllLoader
    from utils import get_bigrams
    import os
    from fastNLP import Vocabulary
    from fastNLP.embeddings import StaticEmbedding
    from fastNLP.io.data_bundle import DataBundle

    loader = ConllLoader(['chars','target'])
    # bundle = loader.load(path)
    #
    # datasets = bundle.datasets

    # print(datasets['train'][:5])

    train_path = os.path.join(path,'train.conllx')
    dev_path = os.path.join(path, 'dev.conllx')
    test_path = os.path.join(path, 'test.conllx')

    paths = {}
    paths['train'] = train_path
    paths['dev'] = dev_path
    paths['test'] = test_path

    datasets = {}

    for k,v in paths.items():
        bundle = loader.load(v)
        datasets[k] = bundle.datasets['train']



    for k,v in datasets.items():
        print('{}:{}'.format(k,len(v)))
    # print(*list(datasets.keys()))
    vocabs = {}
    char_vocab = Vocabulary()
    bigram_vocab = Vocabulary()
    label_vocab = Vocabulary(padding=None,unknown=None)

    for k,v in datasets.items():
        # ignore the word segmentation tag
        v.apply_field(lambda x: [w[0] for w in x],'chars','chars')
        v.apply_field(get_bigrams,'chars','bigrams')


    char_vocab.from_dataset(datasets['train'],field_name='chars',no_create_entry_dataset=[datasets['dev'],datasets['test']])
    # label_vocab.from_dataset(datasets['train'],field_name='target')
    if 'ctb9' in path:
        label_vocab.from_dataset(datasets['train'],datasets['dev'],field_name='target')
    elif 'ctb7' in path:
        label_vocab.from_dataset(datasets['train'], datasets['dev'],datasets['test'], field_name='target')
    else:
        label_vocab.from_dataset(datasets['train'],field_name='target')
    print('label_vocab:{}\n{}'.format(len(label_vocab),label_vocab.idx2word))


    for k,v in datasets.items():
        # v.set_pad_val('target',-100)
        v.add_seq_len('chars',new_field_name='seq_len')


    vocabs['words'] = char_vocab
    vocabs['target'] = label_vocab


    bigram_vocab.from_dataset(datasets['train'],field_name='bigrams',no_create_entry_dataset=[datasets['dev'],datasets['test']])
    if index_token:
        for k,v in datasets.items():
            v.rename_field('chars','raw_chars')
            v.rename_field('bigrams','raw_bigrams')
        char_vocab.index_dataset(*list(datasets.values()), field_name='raw_chars', new_field_name='chars')
        bigram_vocab.index_dataset(*list(datasets.values()),field_name='raw_bigrams',new_field_name='bigrams')
        label_vocab.index_dataset(*list(datasets.values()), field_name='target', new_field_name='target')

    # for k,v in datasets.items():
    #     v.set_input('chars','bigrams','seq_len','target')
    #     v.set_target('target','seq_len')

    vocabs['bigram'] = bigram_vocab


    embeddings = {}

    if char_embedding_path is not None:
        unigram_embedding = StaticEmbedding(char_vocab, model_dir_or_name=char_embedding_path,
                                            word_dropout=char_word_dropout,
                                            min_freq=char_min_freq,only_train_min_freq=only_train_min_freq,normalize=norm_embed)
        embeddings['char'] = unigram_embedding

    if bigram_embedding_path is not None:
        bigram_embedding = StaticEmbedding(bigram_vocab, model_dir_or_name=bigram_embedding_path,
                                           word_dropout=0.01,
                                           min_freq=bigram_min_freq,only_train_min_freq=only_train_min_freq,normalize=norm_embed)
        embeddings['bigram'] = bigram_embedding

    for k,v in datasets.items():
        v.rename_field('chars','words')
        v.rename_field('raw_chars', 'raw_words')
    bundle = DataBundle()
    bundle.datasets = datasets
    bundle.vocabs = vocabs
    bundle.embeddings = embeddings
    from utils import transform_bmeso_bundle_to_bio,transform_bio_bundle_to_bioes
    if encoding_type == 'bmeso':
        bundle = transform_bio_bundle_to_bioes(bundle)
        return bundle
        pass
    elif encoding_type == 'bio':
        # bundle = transform_bmeso_bundle_to_bio(bundle)
        return bundle
        pass
    elif encoding_type == 'bioes':
        bundle = transform_bio_bundle_to_bioes(bundle)
        return bundle
        pass

# @cache_results(_cache_fp='cache/msraner1',_refresh=False)
# def load_msra_ner_1(path,char_embedding_path=None,bigram_embedding_path=None,index_token=True,train_clip=True,
#                               char_min_freq=1,bigram_min_freq=1,only_train_min_freq=0):
#     from fastNLP.io.loader import ConllLoader
#     from utils import get_bigrams
#     if train_clip:
#         train_path = os.path.join(path, 'train_dev.char.bmes_clip1')
#         test_path = os.path.join(path, 'test.char.bmes_clip1')
#     else:
#         train_path = os.path.join(path,'train_dev.char.bmes')
#         test_path = os.path.join(path,'test.char.bmes')
#
#     loader = ConllLoader(['chars','target'])
#     train_bundle = loader.load(train_path)
#     test_bundle = loader.load(test_path)
#
#
#     datasets = dict()
#     datasets['train'] = train_bundle.datasets['train']
#     datasets['test'] = test_bundle.datasets['train']
#
#
#     datasets['train'].apply_field(get_bigrams,field_name='chars',new_field_name='bigrams')
#     datasets['test'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')
#
#     datasets['train'].add_seq_len('chars')
#     datasets['test'].add_seq_len('chars')
#
#
#     from fastNLP import Vocabulary
#     char_vocab = Vocabulary()
#     bigram_vocab = Vocabulary()
#     label_vocab = Vocabulary(padding=None,unknown=None)
#     print(datasets.keys())
#     # print(len(datasets['dev']))
#     print(len(datasets['test']))
#     print(len(datasets['train']))
#     char_vocab.from_dataset(datasets['train'],field_name='chars',
#                             no_create_entry_dataset=[datasets['test']] )
#     bigram_vocab.from_dataset(datasets['train'],field_name='bigrams',
#                               no_create_entry_dataset=[datasets['test']])
#     label_vocab.from_dataset(datasets['train'],field_name='target')
#     if index_token:
#         char_vocab.index_dataset(datasets['train'],datasets['test'],
#                                  field_name='chars',new_field_name='chars')
#         bigram_vocab.index_dataset(datasets['train'],datasets['test'],
#                                  field_name='bigrams',new_field_name='bigrams')
#         label_vocab.index_dataset(datasets['train'],datasets['test'],
#                                  field_name='target',new_field_name='target')
#
#     vocabs = {}
#     vocabs['char'] = char_vocab
#     vocabs['label'] = label_vocab
#     vocabs['bigram'] = bigram_vocab
#     vocabs['label'] = label_vocab
#
#     embeddings = {}
#     if char_embedding_path is not None:
#         char_embedding = StaticEmbedding(char_vocab,char_embedding_path,word_dropout=0.01,
#                                          min_freq=char_min_freq,only_train_min_freq=only_train_min_freq)
#         embeddings['char'] = char_embedding
#
#     if bigram_embedding_path is not None:
#         bigram_embedding = StaticEmbedding(bigram_vocab,bigram_embedding_path,word_dropout=0.01,
#                                            min_freq=bigram_min_freq,only_train_min_freq=only_train_min_freq)
#         embeddings['bigram'] = bigram_embedding
#
#     return datasets,vocabs,embeddings

@cache_results(_cache_fp='cache/msra_ner',_refresh=False)
def load_msra_ner(path,char_embedding_path=None,bigram_embedding_path=None,index_token=True,
                       char_min_freq=1,bigram_min_freq=1,only_train_min_freq=0,norm_embed=False,encoding_type='bmeso',train_clip=True):
    assert encoding_type in ['bmeso','bioes','bio']
    from fastNLP.io.loader import ConllLoader
    from utils import get_bigrams
    import os
    from fastNLP import Vocabulary
    from fastNLP.embeddings import StaticEmbedding
    from fastNLP.io.data_bundle import DataBundle

    if train_clip:
        train_path = os.path.join(path, 'train_dev.char.bmes_clip1')
        test_path = os.path.join(path, 'test.char.bmes_clip1')
    else:
        train_path = os.path.join(path,'train_dev.char.bmes')
        test_path = os.path.join(path,'test.char.bmes')

    loader = ConllLoader(['chars','target'])
    train_bundle = loader.load(train_path)
    # dev_bundle = loader.load(dev_path)
    test_bundle = loader.load(test_path)


    datasets = dict()
    datasets['train'] = train_bundle.datasets['train']
    # datasets['dev'] = dev_bundle.datasets['train']
    datasets['test'] = test_bundle.datasets['train']


    datasets['train'].apply_field(get_bigrams,field_name='chars',new_field_name='bigrams')
    # datasets['dev'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')
    datasets['test'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')

    datasets['train'].add_seq_len('chars')
    # datasets['dev'].add_seq_len('chars')
    datasets['test'].add_seq_len('chars')



    char_vocab = Vocabulary()
    bigram_vocab = Vocabulary()
    label_vocab = Vocabulary(padding=None,unknown=None)
    # print(datasets.keys())
    # print(len(datasets['dev']))
    # print(len(datasets['test']))
    # print(len(datasets['train']))
    char_vocab.from_dataset(datasets['train'],field_name='chars',
                            no_create_entry_dataset=[datasets['test']])
    bigram_vocab.from_dataset(datasets['train'],field_name='bigrams',
                              no_create_entry_dataset=[datasets['test']])
    if 'ctb9' in path:
        label_vocab.from_dataset(datasets['train'],datasets['test'],field_name='target')
    else:
        label_vocab.from_dataset(datasets['train'],field_name='target')
    if index_token:
        for k,v in datasets.items():
            v.rename_field('chars','raw_chars')
            v.rename_field('bigrams','raw_bigrams')
        char_vocab.index_dataset(datasets['train'],datasets['test'],
                                 field_name='raw_chars',new_field_name='chars')
        bigram_vocab.index_dataset(datasets['train'],datasets['test'],
                                 field_name='raw_bigrams',new_field_name='bigrams')
        label_vocab.index_dataset(datasets['train'],datasets['test'],
                                 field_name='target',new_field_name='target')

    vocabs = {}
    vocabs['words'] = char_vocab
    # vocabs['label'] = label_vocab
    vocabs['bigrams'] = bigram_vocab
    vocabs['target'] = label_vocab

    embeddings = {}
    if char_embedding_path is not None:
        char_embedding = StaticEmbedding(char_vocab,char_embedding_path,word_dropout=0.01,
                                         min_freq=char_min_freq,only_train_min_freq=only_train_min_freq,normalize=norm_embed)
        embeddings['char'] = char_embedding

    if bigram_embedding_path is not None:
        bigram_embedding = StaticEmbedding(bigram_vocab,bigram_embedding_path,word_dropout=0.01,
                                           min_freq=bigram_min_freq,only_train_min_freq=only_train_min_freq,normalize=norm_embed)
        embeddings['bigram'] = bigram_embedding

    for k,v in datasets.items():
        v.rename_field('chars','words')
        v.rename_field('raw_chars','raw_words')
    bundle = DataBundle()
    bundle.datasets = datasets
    bundle.vocabs = vocabs
    bundle.embeddings = embeddings
    # bundle.datasets['test']
    from utils import transform_bmeso_bundle_to_bio
    if encoding_type == 'bmeso':
        return bundle
    elif encoding_type == 'bio':
        bundle = transform_bmeso_bundle_to_bio(bundle)
        return bundle
    elif encoding_type == 'bioes':
        return bundle



@cache_results(_cache_fp='cache/ud_pos',_refresh=False)
def load_ud_pos(path,char_embedding_path=None,bigram_embedding_path=None,index_token=True,
                       char_min_freq=1,bigram_min_freq=1,only_train_min_freq=0,norm_embed=False,encoding_type='bmeso'):
    assert encoding_type in ['bmeso','bioes','bio']
    from fastNLP.io.loader import ConllLoader
    from utils import get_bigrams
    import os
    from fastNLP import Vocabulary
    from fastNLP.embeddings import StaticEmbedding
    from fastNLP.io.data_bundle import DataBundle

    train_path = os.path.join(path,'train.bmes{}'.format(''))
    dev_path = os.path.join(path,'dev.bmes')
    test_path = os.path.join(path,'test.bmes')

    loader = ConllLoader(['chars','target'])
    train_bundle = loader.load(train_path)
    dev_bundle = loader.load(dev_path)
    test_bundle = loader.load(test_path)


    datasets = dict()
    datasets['train'] = train_bundle.datasets['train']
    datasets['dev'] = dev_bundle.datasets['train']
    datasets['test'] = test_bundle.datasets['train']


    datasets['train'].apply_field(get_bigrams,field_name='chars',new_field_name='bigrams')
    datasets['dev'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')
    datasets['test'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')

    datasets['train'].add_seq_len('chars')
    datasets['dev'].add_seq_len('chars')
    datasets['test'].add_seq_len('chars')



    char_vocab = Vocabulary()
    bigram_vocab = Vocabulary()
    label_vocab = Vocabulary(padding=None,unknown=None)
    # print(datasets.keys())
    # print(len(datasets['dev']))
    # print(len(datasets['test']))
    # print(len(datasets['train']))
    char_vocab.from_dataset(datasets['train'],field_name='chars',
                            no_create_entry_dataset=[datasets['dev'],datasets['test']])
    bigram_vocab.from_dataset(datasets['train'],field_name='bigrams',
                              no_create_entry_dataset=[datasets['dev'],datasets['test']])
    if 'ctb9' in path:
        label_vocab.from_dataset(datasets['train'],datasets['dev'],datasets['test'],field_name='target')
    else:
        label_vocab.from_dataset(datasets['train'],field_name='target')
    if index_token:
        for k,v in datasets.items():
            v.rename_field('chars','raw_chars')
            v.rename_field('bigrams','raw_bigrams')
        char_vocab.index_dataset(datasets['train'],datasets['dev'],datasets['test'],
                                 field_name='raw_chars',new_field_name='chars')
        bigram_vocab.index_dataset(datasets['train'],datasets['dev'],datasets['test'],
                                 field_name='raw_bigrams',new_field_name='bigrams')
        label_vocab.index_dataset(datasets['train'],datasets['dev'],datasets['test'],
                                 field_name='target',new_field_name='target')

    vocabs = {}
    vocabs['words'] = char_vocab
    # vocabs['label'] = label_vocab
    vocabs['bigrams'] = bigram_vocab
    vocabs['target'] = label_vocab

    embeddings = {}
    if char_embedding_path is not None:
        char_embedding = StaticEmbedding(char_vocab,char_embedding_path,word_dropout=0.01,
                                         min_freq=char_min_freq,only_train_min_freq=only_train_min_freq,normalize=norm_embed)
        embeddings['char'] = char_embedding

    if bigram_embedding_path is not None:
        bigram_embedding = StaticEmbedding(bigram_vocab,bigram_embedding_path,word_dropout=0.01,
                                           min_freq=bigram_min_freq,only_train_min_freq=only_train_min_freq,normalize=norm_embed)
        embeddings['bigram'] = bigram_embedding

    for k,v in datasets.items():
        v.rename_field('chars','words')
        v.rename_field('raw_chars','raw_words')
    bundle = DataBundle()
    bundle.datasets = datasets
    bundle.vocabs = vocabs
    bundle.embeddings = embeddings
    from utils import transform_bmeso_bundle_to_bio
    if encoding_type == 'bmeso':
        return bundle
    elif encoding_type == 'bio':
        bundle = transform_bmeso_bundle_to_bio(bundle)
        return bundle
    elif encoding_type == 'bioes':
        return bundle

@cache_results(_cache_fp='cache/ud_seg',_refresh=False)
def load_ud_seg(path,char_embedding_path=None,bigram_embedding_path=None,index_token=True,
                       char_min_freq=1,bigram_min_freq=1,only_train_min_freq=0,norm_embed=False,encoding_type='bmeso'):
    assert encoding_type in ['bmeso','bioes','bio']
    from fastNLP.io.loader import ConllLoader
    from utils import get_bigrams
    import os
    from fastNLP import Vocabulary
    from fastNLP.embeddings import StaticEmbedding
    from fastNLP.io.data_bundle import DataBundle

    train_path = os.path.join(path,'train.bmes{}'.format(''))
    dev_path = os.path.join(path,'dev.bmes')
    test_path = os.path.join(path,'test.bmes')

    loader = ConllLoader(['chars','target'])
    train_bundle = loader.load(train_path)
    dev_bundle = loader.load(dev_path)
    test_bundle = loader.load(test_path)


    datasets = dict()
    datasets['train'] = train_bundle.datasets['train']
    datasets['dev'] = dev_bundle.datasets['train']
    datasets['test'] = test_bundle.datasets['train']

    def transform_pos_into_seg(pos_target):
        seg_target = list(map(lambda x:'{}-SEG'.format(x[0]),pos_target))
        return seg_target
    # DataSet.apply_field()
    for k,v in datasets.items():
        v.apply_field(transform_pos_into_seg,'target','target')


    datasets['train'].apply_field(get_bigrams,field_name='chars',new_field_name='bigrams')
    datasets['dev'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')
    datasets['test'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')

    datasets['train'].add_seq_len('chars')
    datasets['dev'].add_seq_len('chars')
    datasets['test'].add_seq_len('chars')




    char_vocab = Vocabulary()
    bigram_vocab = Vocabulary()
    label_vocab = Vocabulary(padding=None,unknown=None)
    # print(datasets.keys())
    # print(len(datasets['dev']))
    # print(len(datasets['test']))
    # print(len(datasets['train']))
    char_vocab.from_dataset(datasets['train'],field_name='chars',
                            no_create_entry_dataset=[datasets['dev'],datasets['test']])
    bigram_vocab.from_dataset(datasets['train'],field_name='bigrams',
                              no_create_entry_dataset=[datasets['dev'],datasets['test']])
    if 'ctb9' in path:
        label_vocab.from_dataset(datasets['train'],datasets['dev'],datasets['test'],field_name='target')
    else:
        label_vocab.from_dataset(datasets['train'],field_name='target')
    if index_token:
        for k,v in datasets.items():
            v.rename_field('chars','raw_chars')
            v.rename_field('bigrams','raw_bigrams')
        char_vocab.index_dataset(datasets['train'],datasets['dev'],datasets['test'],
                                 field_name='raw_chars',new_field_name='chars')
        bigram_vocab.index_dataset(datasets['train'],datasets['dev'],datasets['test'],
                                 field_name='raw_bigrams',new_field_name='bigrams')
        label_vocab.index_dataset(datasets['train'],datasets['dev'],datasets['test'],
                                 field_name='target',new_field_name='target')

    vocabs = {}
    vocabs['words'] = char_vocab
    # vocabs['label'] = label_vocab
    vocabs['bigrams'] = bigram_vocab
    vocabs['target'] = label_vocab

    embeddings = {}
    if char_embedding_path is not None:
        char_embedding = StaticEmbedding(char_vocab,char_embedding_path,word_dropout=0.01,
                                         min_freq=char_min_freq,only_train_min_freq=only_train_min_freq,normalize=norm_embed)
        embeddings['char'] = char_embedding

    if bigram_embedding_path is not None:
        bigram_embedding = StaticEmbedding(bigram_vocab,bigram_embedding_path,word_dropout=0.01,
                                           min_freq=bigram_min_freq,only_train_min_freq=only_train_min_freq,normalize=norm_embed)
        embeddings['bigram'] = bigram_embedding

    for k,v in datasets.items():
        v.rename_field('chars','words')
        v.rename_field('raw_chars','raw_words')
    bundle = DataBundle()
    bundle.datasets = datasets
    bundle.vocabs = vocabs
    bundle.embeddings = embeddings
    from utils import transform_bmeso_bundle_to_bio
    if encoding_type == 'bmeso':
        return bundle
    elif encoding_type == 'bio':
        bundle = transform_bmeso_bundle_to_bio(bundle)
        return bundle
    elif encoding_type == 'bioes':
        return bundle

@cache_results(_cache_fp='cache/ctb', _refresh=True)
def load_ctb_seg(path,char_embedding_path=None,bigram_embedding_path=None,index_token=True,
                   char_min_freq=1,bigram_min_freq=1,only_train_min_freq=0,char_word_dropout=0.01,encoding_type='bio',norm_embed=True):
    from fastNLP.io.loader import ConllLoader
    from utils import get_bigrams

    assert encoding_type in ['bmeso','bioes','bio']
    from fastNLP.io.loader import ConllLoader
    from utils import get_bigrams
    import os
    from fastNLP import Vocabulary
    from fastNLP.embeddings import StaticEmbedding
    from fastNLP.io.data_bundle import DataBundle

    loader = ConllLoader(['chars','target'])
    # bundle = loader.load(path)
    #
    # datasets = bundle.datasets

    # print(datasets['train'][:5])

    train_path = os.path.join(path,'train.conllx')
    dev_path = os.path.join(path, 'dev.conllx')
    test_path = os.path.join(path, 'test.conllx')

    paths = {}
    paths['train'] = train_path
    paths['dev'] = dev_path
    paths['test'] = test_path

    datasets = {}

    for k,v in paths.items():
        bundle = loader.load(v)
        datasets[k] = bundle.datasets['train']

    def transform_pos_into_seg(pos_target):
        seg_target = list(map(lambda x:'{}-SEG'.format(x[0]),pos_target))
        return seg_target
    # DataSet.apply_field()
    for k,v in datasets.items():
        v.apply_field(transform_pos_into_seg,'target','target')



    for k,v in datasets.items():
        print('{}:{}'.format(k,len(v)))
    # print(*list(datasets.keys()))
    vocabs = {}
    char_vocab = Vocabulary()
    bigram_vocab = Vocabulary()
    label_vocab = Vocabulary(padding=None,unknown=None)

    for k,v in datasets.items():
        # ignore the word segmentation tag
        v.apply_field(lambda x: [w[0] for w in x],'chars','chars')
        v.apply_field(get_bigrams,'chars','bigrams')


    char_vocab.from_dataset(datasets['train'],field_name='chars',no_create_entry_dataset=[datasets['dev'],datasets['test']])
    # label_vocab.from_dataset(datasets['train'],field_name='target')
    if 'ctb9' in path:
        label_vocab.from_dataset(datasets['train'],datasets['dev'],field_name='target')
    elif 'ctb7' in path:
        label_vocab.from_dataset(datasets['train'], datasets['dev'],datasets['test'], field_name='target')
    else:
        label_vocab.from_dataset(datasets['train'],field_name='target')
    print('label_vocab:{}\n{}'.format(len(label_vocab),label_vocab.idx2word))


    for k,v in datasets.items():
        # v.set_pad_val('target',-100)
        v.add_seq_len('chars',new_field_name='seq_len')


    vocabs['words'] = char_vocab
    vocabs['target'] = label_vocab


    bigram_vocab.from_dataset(datasets['train'],field_name='bigrams',no_create_entry_dataset=[datasets['dev'],datasets['test']])
    if index_token:
        for k,v in datasets.items():
            v.rename_field('chars','raw_chars')
            v.rename_field('bigrams','raw_bigrams')
        char_vocab.index_dataset(*list(datasets.values()), field_name='raw_chars', new_field_name='chars')
        bigram_vocab.index_dataset(*list(datasets.values()),field_name='raw_bigrams',new_field_name='bigrams')
        label_vocab.index_dataset(*list(datasets.values()), field_name='target', new_field_name='target')

    # for k,v in datasets.items():
    #     v.set_input('chars','bigrams','seq_len','target')
    #     v.set_target('target','seq_len')

    vocabs['bigram'] = bigram_vocab


    embeddings = {}

    if char_embedding_path is not None:
        unigram_embedding = StaticEmbedding(char_vocab, model_dir_or_name=char_embedding_path,
                                            word_dropout=char_word_dropout,
                                            min_freq=char_min_freq,only_train_min_freq=only_train_min_freq,normalize=norm_embed)
        embeddings['char'] = unigram_embedding

    if bigram_embedding_path is not None:
        bigram_embedding = StaticEmbedding(bigram_vocab, model_dir_or_name=bigram_embedding_path,
                                           word_dropout=0.01,
                                           min_freq=bigram_min_freq,only_train_min_freq=only_train_min_freq,normalize=norm_embed)
        embeddings['bigram'] = bigram_embedding

    for k,v in datasets.items():
        v.rename_field('chars','words')
        v.rename_field('raw_chars', 'raw_words')
    bundle = DataBundle()
    bundle.datasets = datasets
    bundle.vocabs = vocabs
    bundle.embeddings = embeddings
    from utils import transform_bmeso_bundle_to_bio,transform_bio_bundle_to_bioes
    if encoding_type == 'bmeso':
        bundle = transform_bio_bundle_to_bioes(bundle)
        return bundle
        pass
    elif encoding_type == 'bio':
        # bundle = transform_bmeso_bundle_to_bio(bundle)
        return bundle
        pass
    elif encoding_type == 'bioes':
        bundle = transform_bio_bundle_to_bioes(bundle)
        return bundle
        pass

@cache_results(_cache_fp='cache/clue_ner', _refresh=True)
def load_clue_ner(path,unigram_embedding_path=None,bigram_embedding_path=None,index_token=True,
                   char_min_freq=1,bigram_min_freq=1,only_train_min_freq=0,char_word_dropout=0.01,encoding_type='bio',norm_embed=True):
    from fastNLP.io.loader import ConllLoader
    from utils import get_bigrams

    assert encoding_type in ['bmeso','bioes','bio']
    from fastNLP.io.loader import ConllLoader
    from utils import get_bigrams
    import os
    from fastNLP import Vocabulary
    from fastNLP.embeddings import StaticEmbedding
    from fastNLP.io.data_bundle import DataBundle

    loader = ConllLoader(['chars','target'])
    # bundle = loader.load(path)
    #
    # datasets = bundle.datasets

    # print(datasets['train'][:5])

    train_path = os.path.join(path,'train.conll')
    dev_path = os.path.join(path, 'dev.conll')
    test_path = os.path.join(path, 'test.conll')

    paths = {}
    paths['train'] = train_path
    paths['dev'] = dev_path
    paths['test'] = test_path

    datasets = {}

    for k,v in paths.items():
        bundle = loader.load(v)
        datasets[k] = bundle.datasets['train']



    for k,v in datasets.items():
        print('{}:{}'.format(k,len(v)))
    # print(*list(datasets.keys()))
    vocabs = {}
    char_vocab = Vocabulary()
    bigram_vocab = Vocabulary()
    label_vocab = Vocabulary(padding=None,unknown=None)

    for k,v in datasets.items():
        # ignore the word segmentation tag
        v.apply_field(lambda x: [w[0] for w in x],'chars','chars')
        v.apply_field(get_bigrams,'chars','bigrams')


    char_vocab.from_dataset(datasets['train'],field_name='chars',no_create_entry_dataset=[datasets['dev'],datasets['test']])
    label_vocab.from_dataset(datasets['train'],field_name='target')
    print('label_vocab:{}\n{}'.format(len(label_vocab),label_vocab.idx2word))


    for k,v in datasets.items():
        # v.set_pad_val('target',-100)
        v.add_seq_len('chars',new_field_name='seq_len')


    vocabs['words'] = char_vocab
    vocabs['target'] = label_vocab


    bigram_vocab.from_dataset(datasets['train'],field_name='bigrams',no_create_entry_dataset=[datasets['dev'],datasets['test']])
    if index_token:
        for k,v in datasets.items():
            v.rename_field('chars','raw_chars')
            v.rename_field('bigrams','raw_bigrams')
        char_vocab.index_dataset(*list(datasets.values()), field_name='raw_chars', new_field_name='chars')
        bigram_vocab.index_dataset(*list(datasets.values()),field_name='raw_bigrams',new_field_name='bigrams')
        label_vocab.index_dataset(*list(datasets.values()), field_name='target', new_field_name='target')

    # for k,v in datasets.items():
    #     v.set_input('chars','bigrams','seq_len','target')
    #     v.set_target('target','seq_len')

    vocabs['bigram'] = bigram_vocab


    embeddings = {}

    if unigram_embedding_path is not None:
        unigram_embedding = StaticEmbedding(char_vocab, model_dir_or_name=unigram_embedding_path,
                                            word_dropout=char_word_dropout,
                                            min_freq=char_min_freq,only_train_min_freq=only_train_min_freq,normalize=norm_embed)
        embeddings['char'] = unigram_embedding

    if bigram_embedding_path is not None:
        bigram_embedding = StaticEmbedding(bigram_vocab, model_dir_or_name=bigram_embedding_path,
                                           word_dropout=0.01,
                                           min_freq=bigram_min_freq,only_train_min_freq=only_train_min_freq,normalize=norm_embed)
        embeddings['bigram'] = bigram_embedding

    for k,v in datasets.items():
        v.rename_field('chars','words')
        v.rename_field('raw_chars','raw_words')
    bundle = DataBundle()
    bundle.datasets = datasets
    bundle.vocabs = vocabs
    bundle.embeddings = embeddings
    from utils import transform_bmeso_bundle_to_bio,transform_bio_bundle_to_bioes
    if encoding_type == 'bmeso':
        bundle = transform_bio_bundle_to_bioes(bundle)
        return bundle
        pass
    elif encoding_type == 'bio':
        # bundle = transform_bmeso_bundle_to_bio(bundle)
        return bundle
        pass
    elif encoding_type == 'bioes':
        bundle = transform_bio_bundle_to_bioes(bundle)
        return bundle
        pass

    # return datasets, vocabs, embeddings



if __name__ == '__main__':
    bundle = load_msra_ner_1(msra_ner_cn_path)
    for k,v in bundle.datasets.items():
        print('{}:{}'.format(k,len(v)))
    exit()
    # bundle = load_conll(conll_path)
    # print(bundle.datasets['dev'])
    # bundle = load_ecom_ner(ecom_ner_path,_refresh=True)
    # from fastNLP.io.loader import ConllLoader
    # loader = ecom_conll_loader(['chars','target'],sep='\t',dropna=False)
    # bundle = loader.load('/remote-home/xnli/data/corpus/202111.txt')
    # print(bundle)
    cache_name = 'cache/tmp'
    path_ = ctb7_char_path
    ctb_5_bundle_bio = load_ctb_pos(path_,index_token=True,char_min_freq=1,bigram_min_freq=1,norm_embed=True,
                                char_embedding_path=yangjie_rich_pretrain_unigram_path,bigram_embedding_path=yangjie_rich_pretrain_bigram_path,
                                _cache_fp='{}_bio'.format(path_),_refresh=False,encoding_type='bio')
    for k,v in ctb_5_bundle_bio.datasets.items():
        print('{}:{}'.format(k,len(v)))
    exit()
    # ctb_5_bundle_bioes = load_ctb_pos(path_,index_token=True,char_min_freq=1,bigram_min_freq=1,norm_embed=True,
    #                             char_embedding_path=yangjie_rich_pretrain_unigram_path,bigram_embedding_path=yangjie_rich_pretrain_bigram_path,
    #                             _cache_fp='{}_bioes'.format(path_),_refresh=False,encoding_type='bioes')
    #
    # from fastNLP.core.metrics import _bioes_tag_to_spans,_bio_tag_to_spans



    for k,v in ctb_5_bundle_bio.datasets.items():
        print('{}:{}'.format(k,len(v)))
        # print('{}:{}'.format(k,v))

        for i,ins in enumerate(v):
            target_bio = list(map(ctb_5_bundle_bio.vocabs['target'].to_word,ins['target']))
            target_bioes = list(map(ctb_5_bundle_bioes.vocabs['target'].to_word,ctb_5_bundle_bioes.datasets[k][i]['target']))

            span1 = _bio_tag_to_spans(target_bio)
            span2 = _bioes_tag_to_spans(target_bioes)

            assert span1 == span2
