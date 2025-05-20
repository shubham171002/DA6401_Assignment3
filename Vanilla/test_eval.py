# sample_pred.py  – predictions on test set
import torch, wandb, os
from tqdm.auto import tqdm
from data   import get_dataloaders
from models import Encoder, Decoder, Seq2Seq
from train import compute_exact_accuracy
import os, csv, torch, random
from pathlib import Path
from tabulate import tabulate
from vocab import CharVocab


random.seed(42)
BEST_CKPT = "BEST_model.pth"
LANG      = "gu"
BATCH     = 1

device = "cuda" if torch.cuda.is_available() else "cpu"
loaders, src_vocab, tgt_vocab = get_dataloaders(LANG, batch_size=BATCH, device=device)

# 1. load model
enc = Encoder(src_vocab.size, BEST_CONFIG["emb_size"],
              BEST_CONFIG["hidden_size"], BEST_CONFIG["layers"],
              BEST_CONFIG["cell"],  BEST_CONFIG["dropout"]).to(device)

dec = Decoder(tgt_vocab.size, BEST_CONFIG["emb_size"],
              BEST_CONFIG["hidden_size"],
              BEST_CONFIG["layers"], BEST_CONFIG["cell"],
              BEST_CONFIG["dropout"]).to(device)

model = Seq2Seq(enc, dec, pad_idx=src_vocab.pad_idx, device=device).to(device)
model.load_state_dict(torch.load(BEST_CKPT, map_location=device))
model.eval()

# 2. accuracy on test set
correct = total = 0
pred_rows = []
all_samples = []  # collect all for random sampling later

with torch.no_grad():
    for i, (src, src_len, tgt) in enumerate(loaders["test"]):
        src, src_len, tgt = (x.to(device) for x in (src, src_len, tgt))

        pred = model.infer_greedy(src, src_len, tgt_vocab, max_len=tgt.size(1))
        pred_str  = tgt_vocab.decode(pred[0].cpu(), strip_specials=True)
        gold_str  = tgt_vocab.decode(tgt[0, 1:].cpu(), strip_specials=True)
        input_str = src_vocab.decode(src[0].cpu(), strip_specials=True)
        is_corr   = pred_str == gold_str
        
        correct += int(is_corr)
        total   += 1

        row = (i, input_str, gold_str, pred_str, is_corr)
        pred_rows.append(row)
        all_samples.append(row)

test_acc = correct / total
print(f"\nExact-match TEST accuracy: {test_acc*100:.2f}% ({correct}/{total})\n")

# 3. 20 random-row sample table
random.seed(42)  # ensures reproducibility of sample
sample_rows = random.sample(all_samples, 20)

headers = ["idx", "input (src)", "gold (tgt)", "prediction", "correct"]
print(tabulate(sample_rows, headers=headers, tablefmt="github"))

# 4. save ALL test predictions 
out_dir = Path("predictions_vanilla"); out_dir.mkdir(exist_ok=True)
out_file = out_dir / "vanilla_predictions.tsv"
with out_file.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(headers)           # header row
    writer.writerows(pred_rows)

print(f"\nSaved {len(pred_rows)} predictions to {out_file.absolute()}")