# data.py

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from vocab import CharVocab
import os

def read_tsv(path):
    with open(path, encoding='utf-8') as f:
        for ln in f:
            parts = ln.strip().split('\t')
            if len(parts) >= 2:
                yield parts[1], parts[0]

class Seq2SeqDataset(Dataset):
    def __init__(self, path, src_vocab, tgt_vocab):
        self.examples = []
        for src, tgt in read_tsv(path):
            src_ids = src_vocab.encode(src, add_sos=True, add_eos=True)
            tgt_ids = tgt_vocab.encode(tgt, add_sos=True, add_eos=True)
            self.examples.append((torch.tensor(src_ids, dtype=torch.long),
                                   torch.tensor(tgt_ids, dtype=torch.long)))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def collate_fn(batch, src_vocab, tgt_vocab):
    srcs, tgts = zip(*batch)
    srcs_p = pad_sequence(srcs, batch_first=True, padding_value=src_vocab.pad_idx)
    tgts_p = pad_sequence(tgts, batch_first=True, padding_value=tgt_vocab.pad_idx)
    src_lens = torch.tensor([len(s) for s in srcs], dtype=torch.long)
    return srcs_p, src_lens, tgts_p



def get_dataloaders(
        lang: str = 'gu',
        batch_size: int = 64,
        device: str = 'cpu',         
        num_workers: int = 2,       
        prefetch_factor: int = 4,   
        persistent_workers: bool = True 
    ):
    
    # Build train / dev / test DataLoaders for the Dakshina transliteration task.
    base = os.path.join(
        'data/dakshina_dataset_v1.0',
        lang, 'lexicons'
    )

    # build vocabularies on train + dev
    all_src, all_tgt = [], []
    for split in ['train', 'dev']:
        path = os.path.join(base, f'{lang}.translit.sampled.{split}.tsv')
        for s, t in read_tsv(path):
            all_src.append(s)
            all_tgt.append(t)

    src_vocab = CharVocab.build_from_texts(all_src)
    tgt_vocab = CharVocab.build_from_texts(all_tgt)

    # common DataLoader kwargs
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        pin_memory=(device == 'cuda')          # True only when enabled GPU
    )

    loaders = {}
    for split in ['train', 'dev', 'test']:
        path = os.path.join(base, f'{lang}.translit.sampled.{split}.tsv')
        ds = Seq2SeqDataset(path, src_vocab, tgt_vocab)
        loaders[split] = DataLoader(
            ds,
            shuffle=(split == 'train'),
            collate_fn=lambda b: collate_fn(b, src_vocab, tgt_vocab),
            **loader_kwargs
        )

    return loaders, src_vocab, tgt_vocab