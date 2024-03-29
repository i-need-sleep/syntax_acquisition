import pathlib
import argparse
import json
import os
import random

import transformers

import utils.globals as globals
import utils.train_tokenizer
import utils.dataset

def train(args):
    print(args)

    # Retrieve a list of paths for the training/dev data
    # If loading from a config
    if args.load_config != '':
        config_save_path = f'{globals.CONFIG_DIR}/{args.load_config}.json'
        print(f'Loading config: {config_save_path}')
        args.dataset, args.name = args.load_config.split('-')
        save_name = f'{args.dataset}-{args.name}'
        with open(config_save_path, 'r') as f:
            loaded = json.load(f)
            train_data_paths, dev_data_paths = loaded['train_data_paths'], loaded['dev_data_paths']
    else:
        save_name = f'{args.dataset}-{args.name}'
        if args.dataset == 'babylm_100M':
            train_data_paths = [str(x) for x in pathlib.Path(f'{utils.globals.DATA_DIR}/babylm_data/babylm_100M').glob(f'*.train')]
            dev_data_paths = [str(x) for x in pathlib.Path(f'{utils.globals.DATA_DIR}/babylm_data/babylm_dev').glob(f'*.dev')]
        elif args.dataset == 'openwebtext':
            # sample k subsets from openwebtext
            data_paths = [str(x) for x in pathlib.Path(f'{utils.globals.DATA_DIR}/openwebtext').glob(f'*.xz')]
            data_paths = random.sample(data_paths, 320)
            train_data_paths, dev_data_paths = data_paths[: len(data_paths)//10*9], data_paths[len(data_paths)//10*9:]
        elif args.dataset == 'subsample_vocab':
            if '0' in args.name:
                run_idx = '_0'
            elif '1' in args.name:
                run_idx = '_1'
            else:
                run_idx = ''
            print(f'run_idx: {run_idx}')
            data_path = f'{utils.globals.DATA_DIR}/subsampled/vocab{run_idx}.txt'
            with open(data_path, 'r') as f:
                text = f.read()
            cutoff = 350000000
            text_train, text_dev = text[: cutoff], text[cutoff: int(cutoff * 1.1)]
            train_path = f'{utils.globals.DATA_DIR}/subsampled/vocab_train{run_idx}.txt'
            dev_path = f'{utils.globals.DATA_DIR}/subsampled/vocab_dev{run_idx}.txt'
            with open(train_path, 'w') as f:
                f.write(text_train)
            with open(dev_path, 'w') as f:
                f.write(text_dev)
            train_data_paths, dev_data_paths = [train_path], [dev_path]
        elif args.dataset == 'subsample_sent_len':
            data_path = f'{utils.globals.DATA_DIR}/subsampled/sent_len.txt'
            with open(data_path, 'r') as f:
                text = f.read()
            text_train, text_dev = text[: -500], text[-500: ]
            train_path = f'{utils.globals.DATA_DIR}/subsampled/sent_len_train.txt'
            dev_path = f'{utils.globals.DATA_DIR}/subsampled/sent_len_dev.txt'
            with open(train_path, 'w') as f:
                f.write(text_train)
            with open(dev_path, 'w') as f:
                f.write(text_dev)
            train_data_paths, dev_data_paths = [train_path], [dev_path]
        else:
            raise NotImplementedError
        
        # Save the selected paths
        config_save_path = f'{globals.CONFIG_DIR}/{save_name}.json'
        print(f'Saving config: {config_save_path}')
        with open(config_save_path, 'w') as f:
            json.dump({
                'train_data_paths': train_data_paths,
                'dev_data_paths': dev_data_paths,
            }, f)

    xz = args.dataset == 'openwebtext'
    
    # Retrieve the tokenizer, or train a new one
    tokenizer_path = f'{globals.TOKENIZER_CHECKPOINT_DIR}/{save_name}'
    if os.path.exists(tokenizer_path):
        print(f'Loading tokenizer: {tokenizer_path}')
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path, xz=xz)
    else:
        print(f'Training tokenizer: {tokenizer_path}')
        tokenizer = utils.train_tokenizer.train_tokenizer(train_data_paths, save_name, xz=xz)

    # Make datasets
    train_set = utils.dataset.LMDataset(train_data_paths, tokenizer, xz=xz)
    dev_set = utils.dataset.LMDataset(dev_data_paths, tokenizer, xz=xz)
    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Initialize the model
    config = transformers.AutoConfig.from_pretrained(
        "distilgpt2",
        vocab_size=len(tokenizer),
        n_ctx=train_set.context_length,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    model = transformers.GPT2LMHeadModel(config)

    # Set up the trainer
    checkpoint_dir = f'{globals.MODEL_CHECKPOINT_DIR}/{save_name}'
    trainer_args = transformers.TrainingArguments(
        output_dir=checkpoint_dir,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        evaluation_strategy="steps",
        eval_steps=5_000,
        logging_steps=5_000,
        gradient_accumulation_steps=8,
        num_train_epochs=50,
        weight_decay=0.1,
        warmup_steps=1_000,
        lr_scheduler_type="cosine",
        learning_rate=5e-4,
        save_steps=5_000,
        fp16=True,
        push_to_hub=False,
    )

    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        args=trainer_args,
        data_collator=data_collator,
        train_dataset=train_set,
        eval_dataset=dev_set,
    )

    trainer.train()
    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--name', default='unnamed')

    # Data
    parser.add_argument('--load_config', default='', type=str) # Load a generated config...
    parser.add_argument('--dataset', default='', type=str) # Or generate a config at {dataset}_{name}

    args = parser.parse_args()

    train(args)