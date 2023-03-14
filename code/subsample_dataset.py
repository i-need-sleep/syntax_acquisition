import argparse
import json
import pathlib
import random
import lzma
import tqdm

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
        n_char += len(text)
        sents = utils.data_utils.split_sents(text)
        if args.debug:
            sents = sents[:10]
        
        for sent in sents:
            sent_vocab = utils.data_utils.get_vocab(sent)
            for v in sent_vocab:
                if v not in out:
                    out.append(v)
    return out, n_char

def subsample_vocab(vocab, tar_n_char, args):
    # Subsample openwebtext until target # char is reached

    out = ''
    data_paths = [str(x) for x in pathlib.Path(f'{utils.globals.DATA_DIR}/openwebtext').glob(f'*.xz')]
    
    while len(out) < tar_n_char:
        # Sample a xz file
        sampled_path = random.sample(data_paths, 1)[0]
        data_paths.pop(data_paths.index(sampled_path))
        
        data_str = ''
        with lzma.open(sampled_path, 'r') as f:
            for line in f:
                line_text = line.decode('UTF-8').strip()
                if line_text != '\n':
                    data_str += f'{line_text} '
        
        sents = utils.data_utils.split_sents(data_str)
        
        if args.debug:
            sents = sents[:10]
        
        # Filter out OOV sents
        for sent in sents: 
            sent_vocab = utils.data_utils.get_vocab(sent)

            skip = False
            for v in sent_vocab:
                if v not in vocab:
                    skip = True

            if skip:
                continue
        
            out += sent + '\n'
    return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--name', default='unnamed')

    # Data
    parser.add_argument('--load_babylm_config', default='', type=str) # The babylm dataset to match
    parser.add_argument('--match_type', default='vocab', type=str) # vocab, sent_len, construct

    args = parser.parse_args()

    subsample(args)