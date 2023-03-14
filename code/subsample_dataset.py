import argparse
import json
import pathlib
import random
import lzma

import tqdm
import nltk
import stanza

import utils.dataset
import utils.data_utils
import utils.globals as globals

def subsample(args):

    if args.debug:
        args.load_babylm_config = 'babylm_100M-poc'

    print(args)

    if args.match_type == 'vocab':
        vocab, n_char = get_vocab_from_config(args.load_babylm_config, args)
        subsample = subsample_vocab(vocab, n_char, args) # one long string
    elif args.match_type == 'sent_len':
        lens = get_sent_len_from_config(args.load_babylm_config, args)
        subsample = subsample_sent_len(lens, args)
    elif args.match_type == 'construct':
        consts = get_consts_from_config(args.load_babylm_config, args)
        subsample = subsample_consts(consts, args)
    else:
        raise NotImplementedError
    
    save_path = f'{globals.DATA_DIR}/subsampled/{args.match_type}.txt'
    print(f'Saving at {save_path}')
    with open(save_path, 'w') as f:
        f.write(subsample)

def get_vocab_from_config(config, args):
    # Fetch a list of uncased words
    out = []
    
    config_path = f'{globals.CONFIG_DIR}/{config}.json'
    with open(config_path, 'r', encoding="utf-8") as f:
        config_data = json.load(f)

    paths = config_data['train_data_paths'] + config_data['dev_data_paths']
    texts = []
    for path in paths:
        with open(path, 'r', encoding="utf-8") as f:
            text = f.read()
            texts.append(text)

    n_char = 0
    for text in tqdm.tqdm(texts):
        if args.debug:
            text = text[:1000]
        n_char += len(text)
        sents = nltk.sent_tokenize(text)
        sents = [nltk.word_tokenize(sent) for sent in sents]
        for sent in sents:
            for v in sent:
                if v not in out:
                    out.append(v)

    return out, n_char

def subsample_vocab(vocab, tar_n_char, args):
    # Subsample openwebtext until target # char is reached

    out = ''
    data_paths = [str(x) for x in pathlib.Path(f'{utils.globals.DATA_DIR}/openwebtext').glob(f'*.xz')]
    
    n_split = 0
    while len(out) < tar_n_char:
        n_split += 1
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
        sents = nltk.sent_tokenize(text)
        sents = [nltk.word_tokenize(sent) for sent in sents]
        for idx, sent in enumerate(sents):
            skip = False
            for v in sent:
                if v not in vocab:
                    skip = True
            if skip:
                continue
            out += ' '.join(sent) + '\n'
    return out

def get_sent_len_from_config(config, args):
    # Fetch a list of lengths
    out = []
    
    config_path = f'{globals.CONFIG_DIR}/{config}.json'
    with open(config_path, 'r', encoding="utf-8") as f:
        config_data = json.load(f)

    paths = config_data['train_data_paths'] + config_data['dev_data_paths']
    texts = []

    if args.debug:
        paths = paths[:2]

    for path in paths:
        with open(path, 'r', encoding="utf-8") as f:
            text = f.read()
            texts.append(text)

    for text in tqdm.tqdm(texts):
        if args.debug:
            text = text[:1000]
        sents = nltk.sent_tokenize(text)
        sents = [nltk.word_tokenize(sent) for sent in sents]
        out.append(len(sents))
    return out

def subsample_sent_len(lens, args):
    # Match sentence length

    out = ''
    data_paths = [str(x) for x in pathlib.Path(f'{utils.globals.DATA_DIR}/openwebtext').glob(f'*.xz')]
    
    n_split = 0
    while len(lens) > 0:
        n_split += 1
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
        sents = nltk.sent_tokenize(text)
        sents = [nltk.word_tokenize(sent) for sent in sents]

        for sent in sents: 
            sent_len = len(sent)
            
            if sent_len in lens:
                out += ' '.join(sent) + '\n'
                lens.pop(lens.index(sent_len))
    return out

def get_consts_from_config(config, args):
    # Fetch a list of uncased words
    consts = []

    # Parser
    parser = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency', package={'constituency': 'wsj_bert'})
    
    config_path = f'{globals.CONFIG_DIR}/{config}.json'
    with open(config_path, 'r', encoding="utf-8") as f:
        config_data = json.load(f)

    paths = config_data['train_data_paths'] + config_data['dev_data_paths']
    texts = []
    for path in paths:
        with open(path, 'r', encoding="utf-8") as f:
            text = f.read()
            texts.append(text)

    for text in tqdm.tqdm(texts):
        if args.debug:
            text = text[:1000]

        doc = parser(text)
        for sent in doc.sentences:
            const = sent.constituency
            const = utils.data_utils.tree_to_zss(const)
            consts.append(str(const))
    return consts

def subsample_consts(consts, args):    
    # Match constructs

    out = ''
    data_paths = [str(x) for x in pathlib.Path(f'{utils.globals.DATA_DIR}/openwebtext').glob(f'*.xz')]
    n_split = 0
    
    while len(consts) > 0:
        n_split += 1
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
            const = sent.constituency
            const = str(utils.data_utils.tree_to_zss(const))
            if const in consts:
                out += ' '.join(sent) + '\n'
                consts.pop(consts.index(const))
        # TODO: soft matching with zss tree edit distance
    return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--name', default='unnamed')

    # Data
    parser.add_argument('--load_babylm_config', default='', type=str) # The babylm dataset to match
    parser.add_argument('--match_type', default='vocab', type=str) # vocab, sent_len, construct

    # Subsampling
    parser.add_argument('--max_n_file', default='5000', type=int) # number of subsets of openwebtext to consider

    args = parser.parse_args()

    subsample(args)