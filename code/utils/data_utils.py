import pickle
import os

import pandas as pd

import utils.globals as globals

def load_colorless_green_rnn():
    df = pd.read_csv(globals.COLORLESSGREENRNN_PATH, sep='\t', header=0)
    cleaned_sents = []
    for i in range(len(df)):
        cleaned_sent = clean_sent(df['sent'][i], df['form'][i], df['prefix'][i])
        cleaned_sent = strip_eos(cleaned_sent)
        cleaned_sents.append(cleaned_sent)
    df['cleaned_sents'] = cleaned_sents
    return df

def strip_eos(sent, eos='<eos>'):
    return sent.split(eos)[0]

def clean_sent(sent, form, prefix):
    postfix = ' '.join(sent[len(prefix):].split(' ')[1:])
    out = f'{prefix} {form} {postfix}'
    return out

def load_snyeval():
    out = {} # {type: {subtype: [correct, incorrect, ...], ...}}

    for file in os.listdir(globals.SYNEVAL_DIR):
        if file[-6:] == 'pickle':
            with open(f'{globals.SYNEVAL_DIR}/{file}', 'rb') as f:
                data = pickle.load(f)
                for key, pairs in data.items():
                    data[key] = flatten_pairs(pairs)
                out[file[: -7]] = data
    
    return out

def flatten_pairs(pairs):
    out = []
    for pair in pairs:
        out += [pair[0], pair[1]]
    return out

def split_sents(text):
    sents = text.replace('\n', ' ').split('.')
    sents = [sent.strip() + '.' for sent in sents[:-1]] + [sents[-1]]
    sents1, out = [], []

    for sent in sents:
        new = sent.split('!')
        new = [n.strip() + '!' for n in new[:-1]] + [new[-1]]
        sents1 += new


    for sent in sents1:
        new = sent.split('?')
        new = [n.strip() + '?' for n in new[:-1]] + [new[-1]]
        out += new
    return out

def get_vocab(sent):
    vocab = []
    cur_word = ''
    for char in sent:
        if char.isalnum():
            cur_word += char
        elif cur_word != '':
            vocab.append(cur_word.lower())
            cur_word = ''
    return vocab


if __name__ == '__main__':
    pass