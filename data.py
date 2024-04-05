import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datasets import load_dataset
from transformers import GPT2TokenizerFast


class DataSet(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()

        self.seq_len = seq_len

        self.num_samples = (data.shape[0]-1) // seq_len

        data = data[:self.num_samples*seq_len+1]

        self.data = [data[i*seq_len:(i+1)*seq_len+1] for i in range(self.num_samples)]

    def __getitem__(self, i):
        return torch.from_numpy(self.data[i].astype(np.int64))
    
    def __len__(self):
        return self.num_samples
    
def tokenize():

    dataset = load_dataset("wikitext",'wikitext-103-raw-v1' )
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    
    val_tokens = []
    for line in tqdm(dataset['validation']['text']):
        line_tokens = tokenizer.encode(line)
        val_tokens.append(np.asarray(line_tokens, dtype=np.uint16))

    val_tokens = np.concatenate(val_tokens)
    np.save('valid.npy', val_tokens)

    test_tokens = []
    for line in tqdm(dataset['test']['text']):
        line_tokens = tokenizer.encode(line)
        test_tokens.append(np.asarray(line_tokens, dtype=np.uint16))

    test_tokens = np.concatenate(test_tokens)
    np.save('test.npy', test_tokens)

    train_tokens = []
    for line in tqdm(dataset['train']['text']):
        line_tokens = tokenizer.encode(line)
        train_tokens.append(np.asarray(line_tokens, dtype=np.uint16))

    train_tokens = np.concatenate(train_tokens)
    np.save('train.npy', train_tokens)


def load_data(path, batch_size, mini_batch_size, seq_len):

    if not os.path.exists(os.path.join(path, 'train.npy')):
        print('dataset not found, start downloading and tokenizing...')
        tokenize()

    train_tokens = np.load(os.path.join(path, 'train.npy'))
    valid_tokens = np.load(os.path.join(path, 'valid.npy'))
    test_tokens = np.load(os.path.join(path, 'test.npy'))

    train_dataset = DataSet(train_tokens, seq_len)
    valid_dataset = DataSet(valid_tokens, seq_len)
    test_dataset = DataSet(test_tokens, seq_len)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                    drop_last=False, pin_memory=True, num_workers=4)
    
    valid_dataloader = DataLoader(valid_dataset, batch_size=mini_batch_size,
                                    drop_last=False, pin_memory=True, num_workers=4)
    
    test_dataloader = DataLoader(test_dataset, batch_size=mini_batch_size,
                                    drop_last=False, pin_memory=True, num_workers=4)
    
    return train_dataloader, valid_dataloader, test_dataloader






