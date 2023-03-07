import torch
import tokenizers

import globals


class LMDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer_dir):
        tokenizer = tokenizers.implementations.ByteLevelBPETokenizer(
            f"{globals.TOKENIZER_CHECKPOINT_DIR}/{tokenizer_dir}/{tokenizer_dir}-vocab.json",
            f"{globals.TOKENIZER_CHECKPOINT_DIR}/{tokenizer_dir}/{tokenizer_dir}-merges.txt",
        )

    def tokenize(self):
        pass

    def __len__(self):
        pass 

    def __getitem__(self, i):
        pass

if __name__ == '__main__':
    pass