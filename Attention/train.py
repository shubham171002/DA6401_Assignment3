# train.py  – with tqdm progress bars
import wandb, torch, torch.nn as nn, torch.optim as optim
from tqdm.auto import tqdm           
from data import get_dataloaders
from models import Encoder, Decoder, Seq2Seq

def compute_exact_accuracy(model, loader, tgt_vocab, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for src, src_lens, tgt in loader:
            src, src_lens, tgt = (x.to(device) for x in (src, src_lens, tgt))
            pred = model.infer_greedy(src, src_lens, tgt_vocab, max_len=tgt.size(1))

            # iterate over the batch
            for b in range(src.size(0)):
                pred_str  = tgt_vocab.decode(pred[b].cpu().tolist())           
                gold_str  = tgt_vocab.decode(tgt[b, 1:].cpu().tolist())   
                correct  += (pred_str == gold_str)
            total += src.size(0)

    return correct / total if total else 0.0


def objective():
    run = wandb.init()
    cfg = run.config
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    loaders, src_vocab, tgt_vocab = get_dataloaders('gu', 
                                                    batch_size=cfg.batch_size,
                                                    device=device)
    
    enc = Encoder(src_vocab.size, cfg.emb_size, cfg.hidden_size,
                  cfg.enc_layers, cfg.cell, cfg.dropout).to(device) 
    dec = Decoder(tgt_vocab.size, cfg.emb_size, cfg.hidden_size, cfg.hidden_size,
                  cfg.enc_layers, cfg.cell, cfg.dropout).to(device)
    model      = Seq2Seq(enc, dec, pad_idx=src_vocab.pad_idx, device=device).to(device)
    criterion  = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_idx)
    optimizer  = optim.Adam(model.parameters(), lr=cfg.lr)

    # epoch loop with tqdm 
    for epoch in tqdm(range(1, cfg.epochs + 1), desc="Epochs", position=0):
        model.train()
        total_loss = 0.0

        # training batches progress-bar
        train_loader = tqdm(loaders['train'], desc=f"Train {epoch}", leave=False, position=1)
        for src, src_lens, tgt in train_loader:
            src, src_lens, tgt = src.to(device), src_lens.to(device), tgt.to(device)

            optimizer.zero_grad()
            output = model(src, src_lens, tgt)
            loss   = criterion(output.reshape(-1, output.size(-1)), tgt[:,1:].reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        train_loader.close()

        train_loss = total_loss / len(loaders['train'])

        # validation loss
        val_loss = 0.0
        val_loader = tqdm(loaders['dev'], desc=f"Val   {epoch}", leave=False, position=1)
        model.eval()
        with torch.no_grad():
            for src, src_lens, tgt in val_loader:
                src, src_lens, tgt = src.to(device), src_lens.to(device), tgt.to(device)
                output  = model(src, src_lens, tgt)
                val_loss += criterion(output.reshape(-1, output.size(-1)),
                                      tgt[:,1:].reshape(-1)).item()
        val_loader.close()
        val_loss /= len(loaders['dev'])

        # accuracy calculation
        train_acc = compute_exact_accuracy(model, loaders['train'], tgt_vocab, device)
        val_acc   = compute_exact_accuracy(model, loaders['dev'],   tgt_vocab, device)

        wandb.log({
            'epoch':      epoch,
            'train_loss': train_loss,
            'val_loss':   val_loss,
            'train_acc':  train_acc,
            'val_acc':    val_acc
        })

    wandb.finish()

if __name__ == "__main__":
    sweep_cfg = {
        'method': 'bayes',
        'metric': {'name': 'val_acc', 'goal': 'maximize'},
        'parameters': {
            'emb_size':    {'values': [128, 256]},
            'hidden_size': {'values': [128, 256, 512]},
            'enc_layers':  {'values': [1, 2]},
            # 'dec_layers':  {'values': [1, 2]},
            'cell':        {'values': ['GRU', 'LSTM']},
            'dropout':     {'values': [0.1, 0.3, 0.5]},
            'lr':          {'values': [8e-4, 5e-4, 1e-4]},
            'batch_size':  {'values': [32, 64]},
            'epochs':      {'value': 20}
        }
    }

    sweep_id = wandb.sweep(
        sweep_cfg,
        entity='da24m020-iit-madras',
        project='DA6401_A3'
    )
    wandb.agent(sweep_id, function=objective, count=100)