import argparse
import json

import utils.dataset
import utils.globals as globals

def subsample(args):

    if args.debug:
        args.load_babylm_config = 'babylm_100M-poc.json'

    print(args)

    if args.match_type == 'vocab':
        vocab = get_vocab_from_config(args.load_babylm_config)
    else:
        raise NotImplementedError

def get_vocab_from_config(config):
    # Fetch a list of uncased words
    
    config_path = f'{globals.CONFIG_DIR}/{config}'
    with open(config_path, 'r') as f:
        config_data = json.load(f)

    paths = config_data['train_data_paths']
    for path in paths:
        with open(path, 'r') as f:
            text = f.read()
            print(text)
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--name', default='unnamed')

    # Data
    parser.add_argument('--load_babylm_config', default='', type=str) # The babylm dataset to match
    parser.add_argument('--load_openwebtext_config', default='', type=str) # The babylm dataset to match
    parser.add_argument('--match_type', default='vocab', type=str) # vocab, sent_len, construct

    args = parser.parse_args()

    subsample(args)