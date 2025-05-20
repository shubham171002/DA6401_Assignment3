# train.py
import wandb, torch, torch.nn as nn, torch.optim as optim
from tqdm.auto import tqdm
from torch.amp import autocast, GradScaler
from data   import get_dataloaders
from models import Encoder, Decoder, Seq2Seq  

torch.backends.cudnn.benchmark = True                 # fastest cuDNN kernels


def compute_exact_accuracy(model, loader, tgt_vocab, device, max_eval=None):
    """String-wise exact match over all or a sampled subset."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for src, src_lens, tgt in loader:
            if max_eval and total >= max_eval: break
            src, src_lens, tgt = (x.to(device) for x in (src, src_lens, tgt))
            pred = model.infer_greedy(src, src_lens, tgt_vocab, max_len=tgt.size(1))
            for b in range(src.size(0)):
                pred_str = tgt_vocab.decode(pred[b].cpu(), strip_specials=True)
                gold_str = tgt_vocab.decode(tgt[b, 1:].cpu(), strip_specials=True)
                correct += (pred_str == gold_str)
            total += src.size(0)
    return correct / max(total, 1)


def objective():
    run  = wandb.init()
    cfg  = run.config
    dev  = 'cuda' if torch.cuda.is_available() else 'cpu'

    loaders, src_v, tgt_v = get_dataloaders(
        'gu', batch_size=cfg.batch_size, device=dev
    )

    enc = Encoder(src_v.size, cfg.emb_size, cfg.hidden_size,
                  layers=cfg.layers, cell=cfg.cell,
                  dropout=cfg.dropout).to(dev)
    dec = Decoder(tgt_v.size, cfg.emb_size, cfg.hidden_size,
                  layers=cfg.layers, cell=cfg.cell,
                  dropout=cfg.dropout).to(dev)
    model = Seq2Seq(enc, dec, pad_idx=src_v.pad_idx, device=dev)
    if torch.__version__.startswith('2'):
        model = torch.compile(model)

    criterion = nn.CrossEntropyLoss(ignore_index=tgt_v.pad_idx)
    opt       = optim.Adam(model.parameters(), lr=cfg.lr)
    scaler    = GradScaler()

    teacher_force = cfg.teacher_force

    for epoch in tqdm(range(1, cfg.epochs + 1), desc="Epochs"):
        model.train(); total_loss = 0
        for src, src_lens, tgt in loaders['train']:
            src, src_lens, tgt = (x.to(dev) for x in (src, src_lens, tgt))
            opt.zero_grad()

            # teacher-forcing mask
            with autocast(device_type='cuda'):
                output = model(src, src_lens, tgt)                  # [B,T-1,V]
                loss   = criterion(output.reshape(-1, output.size(-1)),
                                   tgt[:, 1:].reshape(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            total_loss += loss.item()

        train_loss = total_loss / len(loaders['train'])

        # validation
        val_loss = 0.0; model.eval()
        with torch.no_grad(), autocast(device_type='cuda'):
            for src, src_lens, tgt in loaders['dev']:
                src, src_lens, tgt = (x.to(dev) for x in (src, src_lens, tgt))
                out = model(src, src_lens, tgt)
                val_loss += criterion(out.reshape(-1, out.size(-1)),
                                      tgt[:, 1:].reshape(-1)).item()
        val_loss /= len(loaders['dev'])

        # accuracies (sample 1k train to save time)
        train_acc = compute_exact_accuracy(model, loaders['train'], tgt_v, dev,
                                           max_eval=1000)
        val_acc   = compute_exact_accuracy(model, loaders['dev'],   tgt_v, dev)

        wandb.log({'epoch': epoch,
                   'train_loss': train_loss,
                   'val_loss':   val_loss,
                   'train_acc':  train_acc,
                   'val_acc':    val_acc})

        # early stop for hopeless configs
        if epoch == 5 and val_acc < 0.05:
            run.summary['early_abort'] = True
            return

    run.finish()

sweep_cfg = {
    'method': 'bayes',
    'metric': {'name': 'val_acc', 'goal': 'maximize'},
    'parameters': {
        'emb_size':     {'values': [128, 256, 384]},
        'hidden_size':  {'values': [256, 512, 768]},
        'layers':       {'values': [1, 2, 3]},            # tied encoder = decoder
        'cell':         {'values': ['GRU', 'LSTM','RNN']},  
        'dropout':      {'values': [0.0, 0.2, 0.4]},
        'lr':           {'min': 1e-4, 
                         'max': 8e-4},
        'batch_size':   {'values': [64, 128]},
        'teacher_force':{'values': [0.5, 0.7, 1.0]},  
        'epochs':       {'value': 20}
    }
}

if __name__ == "__main__":
    sweep_id = wandb.sweep(
        sweep_cfg,
        entity='da24m020-iit-madras',
        project='DA6401_A3_no_attn'
    )
    wandb.agent(sweep_id, function=objective, count=100)
