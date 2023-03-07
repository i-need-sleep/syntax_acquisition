from pathlib import Path
from transformers import AutoTokenizer

import utils.globals

def train_tokenizer(paths, name):
    # Define a generator
    def get_corpus():
        for path in paths:
            # TO-DO: Read lines.
            with open(path, 'r', encoding="utf-8") as f:
                yield f.read()
    
    # Load a pre-configured tokenizer
    old_tokenizer = AutoTokenizer.from_pretrained("gpt2")

    tokenizer = old_tokenizer.train_new_from_iterator(get_corpus(), 30000)
    tokenizer.pad_token = tokenizer.eos_token

    # Save files to disk
    save_dir = f'{utils.globals.TOKENIZER_CHECKPOINT_DIR}/{name}'
    tokenizer.save_pretrained(save_dir)
    print(f'Tokenizer saved: {save_dir}')
    return

if __name__ == '__main__':
    # Retrieve the paths. Mind the extention
    paths = [str(x) for x in Path(f'{utils.globals.DATA_DIR}/babylm_100M').glob(f'*.train')]
    train_tokenizer(paths, 'babylm_100M')