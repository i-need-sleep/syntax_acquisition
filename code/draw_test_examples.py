import pathlib
import transformers
from torch.utils.data import DataLoader
import torch
import os
import stanza
import pandas as pd
import tqdm

import utils.dataset
import utils.globals as uglobals

def draw_test_examples():
    # Retrieve the tokenizer, or train a new one
    tokenizer_path = f'{uglobals.TOKENIZER_CHECKPOINT_DIR}/babylm_100M-poc'
    if os.path.exists(tokenizer_path):
        print(f'Loading tokenizer: {tokenizer_path}')
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path, xz=False)

    test_data_paths = [str(x) for x in pathlib.Path(f'{utils.globals.DATA_DIR}/babylm_data/babylm_test').glob(f'*.test')]
    test_set = utils.dataset.LMDataset(test_data_paths, tokenizer, xz=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True)

    parser = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')

    out = []
    for batch_idx, batch in tqdm.tqdm(enumerate(test_loader)):
        if batch_idx > 2000:
            break

        batch = torch.tensor(batch).tolist()
        text = tokenizer.decode(batch)

        doc = parser(text)
        if len(doc.sentences) <= 3:
            continue
        sents = doc.sentences[1: -1]
        for sent in sents:
            out.append(sent.text)
            break

    df = pd.DataFrame({
        'original': out,
        'altered': ['' for _ in out]
    })
    
    save_path = f'{uglobals.DATA_DIR}/babylm_eval/sampled.tsv'
    print(f'Saving at: {save_path}')
    df.to_csv(save_path, sep='\t')

    return

if __name__ == '__main__':
    draw_test_examples()