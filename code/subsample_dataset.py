import argparse
import json
import pathlib
import random
import lzma

import transformers
import tqdm
import nltk
import stanza
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

import utils.dataset
import utils.data_utils
import utils.globals as globals

def subsample(args):

    if args.debug:
        args.load_babylm_config = 'babylm_100M-poc'

    print(args)

    if args.match_type == 'vocab':
        tokenizer, n_char = get_vocab_from_config(args.load_babylm_config, args)
        subsample = subsample_vocab(tokenizer, n_char, args) # one long string
    elif args.match_type == 'sent_len':
        tokenizer, lens = get_sent_len_from_config(args.load_babylm_config, args)
        print(lens)
        if 'openwebtext' in args.config:
            return
        subsample = subsample_sent_len(tokenizer, lens, args)
    elif args.match_type == 'construct':
        consts = get_consts_from_config(args.load_babylm_config, args)
        subsample = subsample_consts(consts, args)
    else:
        raise NotImplementedError
    
    save_path = f'{globals.DATA_DIR}/subsampled/{args.match_type}_{args.name}.txt'
    print(f'Saving at {save_path}, len {len(subsample)}')
    with open(save_path, 'w') as f:
        f.write(subsample)

def get_vocab_from_config(config, args):
    # Fetch a list of uncased words
    
    config_path = f'{globals.CONFIG_DIR}/{config}.json'
    with open(config_path, 'r', encoding="utf-8") as f:
        config_data = json.load(f)

    paths = config_data['train_data_paths'] + config_data['dev_data_paths']
    texts = []
    for path in paths:
        with open(path, 'r', encoding="utf-8") as f:
            text = f.read()
            texts.append(text)

    def get_corpus():
        for t in texts:
            yield t
    
    # Define a word-level tokenizer
    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()])
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    trainer = trainers.WordLevelTrainer(vocab_size=20000, special_tokens=special_tokens)
    tokenizer.train_from_iterator(get_corpus(), trainer=trainer)
    
    n_token = 0
    # for text in texts:
    #     sents = tokenizer.encode_batch(text)
    #     for sent in sents:
    #         n_token += len(sent.tokens)

    return tokenizer, n_token

def subsample_vocab(tokenizer, tar_n_token, args):
    # Subsample openwebtext until target # char is reached

    out = ''
    data_paths = [str(x) for x in pathlib.Path(f'{utils.globals.DATA_DIR}/openwebtext').glob(f'*.xz')]
    
    n_split = 0
    n_token = 0
    while True: # n_token < tar_n_token:
        n_split += 1
        print(n_split)
        if n_split > 1000: #args.max_n_file:
            break
        # Sample a xz file
        sampled_path = random.sample(data_paths, 1)[0]
        data_paths.pop(data_paths.index(sampled_path))
        
        text = ''
        with lzma.open(sampled_path, 'r') as f:
            for line in f:
                line_text = line.decode('UTF-8').strip()
                if line_text != '\n':
                    text += f'{line_text} '
        
        if args.debug:
            text = text[:1000]

        sents = tokenizer.encode_batch(text)
        for sent in sents:
            if not '[UNK]' in sent.tokens:
                out += tokenizer.decode(sent.ids) + '\n'
                n_token += len(sent.tokens)
    return out

def get_sent_len_from_config(config, args):
    # Fetch a list of lengths
    out = {}
    
    config_path = f'{globals.CONFIG_DIR}/{config}.json'
    with open(config_path, 'r', encoding="utf-8") as f:
        config_data = json.load(f)

    save_path = f'{globals.DATA_DIR}/subsampled/sent_len_counts_{args.name}.json'
    tokenizer = stanza.Pipeline(lang='en', processors='tokenize')

    try:
        with open(save_path, 'r') as f:
            out = json.load(f)
        print(f'Loaded from {save_path}')
    except:
        paths = config_data['train_data_paths'] + config_data['dev_data_paths']
        texts = []
        for path in paths:
            if not 'openwebtext' in config:
                with open(path, 'r', encoding="utf-8") as f:
                    text = f.read()
                    texts.append(text)
            else:
                texts = []
                for path in tqdm.tqdm(paths):
                    data_str = ''
                    with lzma.open(path, 'r') as f:
                        for line in f:
                            line_text = line.decode('UTF-8').strip()
                            if line_text != '\n':
                                data_str += f'{line_text} '
                    texts.append(data_str)
            
            for text in texts:
                doc = tokenizer(text)
                for sent in doc.sentences:
                    sent_len = len(sent.words)
                    if sent_len not in out.keys():
                        out[sent_len] = 0
                    out[sent_len] += 1
            
                print(f'Saving at {save_path}')
                with open(save_path, 'w') as f:
                    json.dump(out, f)
        
    return tokenizer, out

def subsample_sent_len(tokenizer, lens, args):
    # Match sentence length

    out = ''
    data_paths = [str(x) for x in pathlib.Path(f'{utils.globals.DATA_DIR}/openwebtext').glob(f'*.xz')]
    
    n_split = 0
    ctr = 0

    for key, val in lens.items():
        if int(key) > 1:
            ctr += val

    while ctr > 0:
        n_split += 1
        print(n_split)
        if n_split > args.max_n_file:
            break
        # Sample a xz file
        sampled_path = random.sample(data_paths, 1)[0]
        data_paths.pop(data_paths.index(sampled_path))
        
        text = ''
        with lzma.open(sampled_path, 'r') as f:
            for idx, line in enumerate(f):
                line_text = line.decode('UTF-8').strip()
                if line_text != '\n':
                    text += f'{line_text} '

        if args.debug:
            text = text[:2000]
            
        doc = tokenizer(text)
        for sent in doc.sentences:
            sent_len = str(len(sent.words))
            if sent_len in lens.keys() and lens[sent_len] > 0:
                out += sent.text + '\n'
                lens[sent_len] -= 1
                ctr -= 1

        save_path = f'{globals.DATA_DIR}/subsampled/{args.match_type}_{args.name}.txt'
        print(f'Saving at {save_path}, len {len(out)}')
        with open(save_path, 'w') as f:
            f.write(out)
    return out

def get_consts_from_config(config, args):
    # Fetch a list of uncased words
    consts = {}

    # Parser
    parser = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
    
    config_path = f'{globals.CONFIG_DIR}/{config}.json'
    with open(config_path, 'r', encoding="utf-8") as f:
        config_data = json.load(f)

    paths = config_data['train_data_paths'] + config_data['dev_data_paths']
    texts = []
    for path in paths:
        with open(path, 'r', encoding="utf-8") as f:
            text = f.read()
            texts.append(text)

    for text_idx, text in tqdm.tqdm(enumerate(texts)):
        print(f'Text: {text_idx}')
        if args.debug:
            text = text[:1000]

        doc = parser(text)
        for sent_idx, sent in enumerate(doc.sentences):
            print(f'Sent: {sent_idx}')
            try:
                const = sent.constituency
                const = str(utils.data_utils.tree_to_zss(const))
                if const not in consts.keys():
                    consts[const] = 0
                consts[const] += 1
            except:
                pass
    
        save_path = f'{globals.DATA_DIR}/subsampled/consts_counts.json'
        print(f'Saving at {save_path}')
        with open(save_path, 'w') as f:
            json.dump(consts, f)
            
    return consts

def subsample_consts(consts, args):    
    # Match constructs

    out = ''
    data_paths = [str(x) for x in pathlib.Path(f'{utils.globals.DATA_DIR}/openwebtext').glob(f'*.xz')]
    n_split = 0

    # Parser
    # TODO: try the BERT-based parser
    parser = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
    
    
    ctr = 0
    for val in consts.values():
        ctr += val

    while ctr > 0:
        n_split += 1
        print(n_split)
        if n_split > args.max_n_file:
            break

        # Sample a xz file
        sampled_path = random.sample(data_paths, 1)[0]
        data_paths.pop(data_paths.index(sampled_path))
        
        text = ''
        with lzma.open(sampled_path, 'r') as f:
            for line in f:
                line_text = line.decode('UTF-8').strip()
                if line_text != '\n':
                    text += f'{line_text} '
        
        if args.debug:
            text = text[:1000]

        doc = parser(text)
        for sent in doc.sentences:
            try:
                const = sent.constituency
                const = str(utils.data_utils.tree_to_zss(const))
                if const in consts.keys() and consts[const] > 0:
                    out += ' '.join(sent) + '\n'
                    consts[const] -= 1
                    ctr -= 1
            except:
                pass
        # TODO: soft matching with zss tree edit distance
    return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--name', default='unnamed')

    # Data
    parser.add_argument('--load_babylm_config', default='babylm_100M-poc', type=str) # The babylm dataset to match
    # parser.add_argument('--load_babylm_config', default='openwebtext-poc', type=str) # The babylm dataset to match
    parser.add_argument('--match_type', default='construct', type=str) # vocab, sent_len, construct

    # Subsampling
    parser.add_argument('--max_n_file', default='2500', type=int) # number of subsets of openwebtext to consider

    args = parser.parse_args()

    subsample(args)