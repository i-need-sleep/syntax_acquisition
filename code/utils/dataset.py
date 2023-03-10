import lzma

import torch
import transformers

import utils.globals as globals


class LMDataset(torch.utils.data.Dataset):
    def __init__(self, data_paths, tokenizer, debug=False, xz=False):
        self.context_length = 128
        self.data_paths = data_paths
        self.tokenizer = tokenizer 
        self.debug = debug
        self.xz = xz
        
        print(f'Tokenizing texts:')
        self.data = self.process()
        print(f'Tokenized #lines: {len(self)}, #tokens: {len(self) * self.context_length}')

    def process(self):
        # Piece all files into a long string
        data_str = ''
        for path in self.data_paths:
            if self.xz:
                with lzma.open(path, 'r') as f:
                    for line in f:
                        line_text = line.decode('UTF-8').strip()
                        if line_text != '\n':
                            data_str += f'{line_text} '
            else:
                with open(path, 'r', encoding="utf-8") as f:
                    for line_text in f.readlines():
                        line_text = line_text.strip()
                        if line_text != '\n':
                            data_str += f'{line_text} '
        
        if self.debug:
            data_str = data_str[:5000]

        # Tokenize into [[ids, ...], ...]
        data = self.tokenize(data_str)
        return data

    def tokenize(self, text):
        # Strings to [[ids, ...], ...]
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.context_length,
            return_overflowing_tokens=True,
            return_length=True,
        )
        
        out = []
        for length, input_ids in zip(tokenized["length"], tokenized["input_ids"]):
            if length == self.context_length:
                out.append(input_ids)

        return out

    def __len__(self):
        return len(self.data) 

    def __getitem__(self, i):
        return self.data[i]
    
# def make_loader(data_paths, tokenizer, batch_size, debug = False):
#     dataset = LMDataset(data_paths, tokenizer, debug)
#     loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=mr_collate, shuffle=True)
#     return loader 

# def mr_collate(data_in):
#     out = torch.tensor(data_in)
#     return out

# transformers.AutoTokenizer.from_pretrained(f'{globals.TOKENIZER_CHECKPOINT_DIR}/{tokenizer_dir}')
