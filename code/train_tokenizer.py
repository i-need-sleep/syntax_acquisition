from pathlib import Path
from tokenizers import ByteLevelBPETokenizer

import utils.globals

def train_tokenizer(input_path, name, extention):
    
    # Retrieve the paths. Mind the extention
    paths = [str(x) for x in Path(f'{utils.globals.DATA_DIR}/{input_path}').glob(f'*.{extention}')]

    # Initialize a tokenizer
    tokenizer = ByteLevelBPETokenizer()

    # Special tokens
    tokenizer.train(files=paths, vocab_size=30_000, min_frequency=2, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])

    # Save files to disk
    save_dir = f'{utils.globals.TOKENIZER_CHECKPOINT_DIR}/{name}'
    tokenizer.save_model(save_dir, name)
    print(f'Saved at {save_dir}')
    
    return

if __name__ == '__main__':
    train_tokenizer('babylm_data/babylm_100M', 'babylm_100M', 'train')