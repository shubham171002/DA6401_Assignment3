import torch, wandb, os
from tqdm.auto import tqdm
from data   import get_dataloaders
from models import Encoder, Decoder, Seq2Seq
from train import compute_exact_accuracy
import os, csv, torch, random
from pathlib import Path
from tabulate import tabulate
from vocab import CharVocab

BEST_CONFIG = dict(
    emb_size      = 384,
    hidden_size   = 512,
    layers        = 3,
    cell          = "LSTM",
    dropout       = 0.4,
    lr            = 0.00010997105521235854,
    batch_size    = 128,
    epochs        = 20,
    teacher_force = 0.5
)

CKPT_OUT = "BEST_model.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
loaders, src_v, tgt_v = get_dataloaders('gu',
                                        batch_size=BEST_CONFIG['batch_size'],
                                        device=device)

enc = Encoder(src_v.size, BEST_CONFIG['emb_size'],
              BEST_CONFIG['hidden_size'], BEST_CONFIG['layers'],
              BEST_CONFIG['cell'], BEST_CONFIG['dropout']).to(device)
dec = Decoder(tgt_v.size, BEST_CONFIG['emb_size'],
              BEST_CONFIG['hidden_size'], BEST_CONFIG['layers'], 
              BEST_CONFIG['cell'], BEST_CONFIG['dropout']).to(device)
model = Seq2Seq(enc, dec, pad_idx=src_v.pad_idx, device=device).to(device)

criterion = torch.nn.CrossEntropyLoss(ignore_index=tgt_v.pad_idx)
opt = torch.optim.Adam(model.parameters(), lr=BEST_CONFIG['lr'])
best_val = 0.0

for epoch in tqdm(range(1, BEST_CONFIG['epochs'] + 1), desc="Epochs"):
    model.train(); tot_loss = 0
    for src, src_len, tgt in loaders['train']:
        src, src_len, tgt = (x.to(device) for x in (src, src_len, tgt))
        opt.zero_grad()
        out = model(src, src_len, tgt)
        loss = criterion(out.reshape(-1, out.size(-1)), tgt[:,1:].reshape(-1))
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); tot_loss += loss.item()

    
    val_acc = compute_exact_accuracy(model, loaders['dev'], tgt_v, device)
    if val_acc > best_val:
        best_val = val_acc
        torch.save(model.state_dict(), CKPT_OUT)
        print(f"[epoch {epoch}] new best val_acc={val_acc:.4f} -> saved {CKPT_OUT}")

print(f"Best val_acc={best_val:.4f}. Checkpoint at {CKPT_OUT}.")
