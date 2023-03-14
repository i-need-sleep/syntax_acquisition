import argparse
import os
import json
import datetime

import evaluate
import pandas as pd

import utils.globals as globals
import utils.data_utils

def eval(args):
    if args.debug:
        args.load_dir = 'babylm_100M-poc'
    print(args)

    # List all checkpoints
    parent_dir = f'{globals.MODEL_CHECKPOINT_DIR}/{args.load_dir}'
    dirs = os.listdir(parent_dir)

    accss = {}

    for dir in dirs:
        d = f'{parent_dir}/{dir}'
        accs = eval_one(d, args.load_dir, dir, args)
        accss[d] = accs
    
    syneval_out_path = f'{globals.OUT_DIR}/{args.load_dir}_syneval.json'
    print(f'Saving syneval acc at {syneval_out_path}')
    with open(syneval_out_path, 'w') as f:
        json.dump(accss, f)
    return

def eval_one(dir, parent, checkpoint, args):

    print(dir)
    perplexity = evaluate.load("./utils/perplexity.py",  module_type= "measurement") #, experiment_id=datetime.datetime.now())

    # Eval SVA
    colorless_df_out, accuracy, test_len = eval_colorless(dir, perplexity, args)
    print(f'SVA accuracy: {accuracy} out of {test_len} pairs')
    save_dir = f'{globals.OUT_DIR}/{parent}/sva/{checkpoint}.csv'
    print(f'Saving at: {save_dir}')
    try:
        os.mkdir(f'{globals.OUT_DIR}/{parent}')
    except:
        pass
    try:
        os.mkdir(f'{globals.OUT_DIR}/{parent}/sva')
    except:
        pass
    colorless_df_out.to_csv(save_dir)

    # Eval LM_syneval
    syneval_out, accuracy, test_len, accs = eval_syneval(dir, perplexity, args)
    print(f'Syneval accuracy: {accuracy} out of {test_len} pairs')
    save_dir = f'{globals.OUT_DIR}/{parent}/syneval/{checkpoint}.csv'
    print(f'Saving at: {save_dir}')
    try:
        os.mkdir(f'{globals.OUT_DIR}/{parent}/syneval')
    except:
        pass
    syneval_out.to_csv(save_dir)

    return accs

def eval_colorless(dir, perplexity, args):
    df = utils.data_utils.load_colorless_green_rnn()
    if args.debug:
        df = df.truncate(after=9)

    sents_in = df['cleaned_sents'].tolist()

    scores = perplexity.compute(data=sents_in, model_id=dir)['perplexities']
    
    preds = []
    n_correct, n_pred = 0, len(scores)/2

    for i in range(0, len(scores), 2):
        if scores[i] > scores[i + 1]:
            preds += [0, 1]
        else:
            preds += [1, 0]
            n_correct += 1

    df['preds'] = preds
    df['perplexity'] = scores
    accuracy = n_correct / n_pred
    return df, accuracy, n_pred

def eval_syneval(dir, perplexity, args):
    data = utils.data_utils.load_snyeval() # {type: {subtype: [correct, incorrect, ...], ...}}
    
    total_n_correct, total_n_pred = 0, 0
    accs = {}
    out = {
        'type': [],
        'subtype': [],
        'sentences': [],
        'prediction': []
    }

    for cons_type, cons_items in data.items():
        accs[cons_type] = {}
        for subtype, sents in cons_items.items():

            scores = perplexity.compute(data=sents, model_id=dir)['perplexities']
    
            preds = []
            n_correct, n_pred = 0, len(scores)/2

            for i in range(0, len(scores), 2):
                if scores[i] > scores[i + 1]:
                    preds += [0, 1]
                else:
                    preds += [1, 0]
                    n_correct += 1

            accuracy = n_correct / n_pred
            print(f'{cons_type} - {subtype} Accuracy: {accuracy} out of {n_pred} pairs')

            # Append to the output
            out['prediction'] += preds
            out['type'] += [cons_type for _ in preds]
            out['subtype'] += [subtype for _ in preds]
            out['sentences'] += sents

            total_n_correct += n_correct
            total_n_pred += n_pred

            accs[cons_type][subtype] = [accuracy, n_pred]

        if args.debug:
            break
    
    df = pd.DataFrame(out)
    accuracy = total_n_correct / total_n_pred
    return df, accuracy, total_n_pred, accs

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true')

    # Data
    parser.add_argument('--load_dir', default='', type=str) # The parent dir for all checkpoints

    args = parser.parse_args()

    eval(args)