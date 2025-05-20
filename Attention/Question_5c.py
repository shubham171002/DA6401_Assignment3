# Question 5 (c) Sample Grid

import matplotlib as mpl

# Path to the local font file
font_path = "Noto_Sans_Gujarati/static/NotoSansGujarati-Regular.ttf"

# Register the font with matplotlib
mpl.font_manager.fontManager.addfont(font_path)
mpl.rcParams["font.family"] = "Noto Sans Gujarati"

import torch, matplotlib.pyplot as plt
from matplotlib import ticker
import random 
import numpy as np
from data   import get_dataloaders
from models import Encoder, Decoder, Seq2Seq

# 1. best hyper-params & checkpoint
BEST_CONFIG = dict(
    emb_size      = 128,
    hidden_size   = 128,
    layers        = 2,
    cell          = "GRU",
    dropout       = 0.5,
    lr            = 1e-4,
    batch_size    = 32,
    epochs        = 20,
    # teacher_force = 0.7
)

random.seed(42)
BEST_CKPT = "BEST_model.pth"
LANG      = "gu"
BATCH     = 1

device = "cuda" if torch.cuda.is_available() else "cpu"
loaders, src_vocab, tgt_vocab = get_dataloaders(LANG, batch_size=BATCH, device=device)

# 2. rebuild best model
enc = Encoder(src_vocab.size, BEST_CONFIG["emb_size"],
              BEST_CONFIG["hidden_size"], BEST_CONFIG["layers"],
              BEST_CONFIG["cell"],  BEST_CONFIG["dropout"]).to(device)

dec = Decoder(tgt_vocab.size, BEST_CONFIG["emb_size"],
              BEST_CONFIG["hidden_size"],BEST_CONFIG['hidden_size'],
              BEST_CONFIG["layers"], BEST_CONFIG["cell"],
              BEST_CONFIG["dropout"]).to(device)

model = Seq2Seq(enc, dec, pad_idx=src_vocab.pad_idx, device=device).to(device)
model.load_state_dict(torch.load(BEST_CKPT, map_location=device))
model.eval()


def infer_with_attn(src, src_len, max_len=50):
    # Returns (generated token ids, attention matrix)

    enc_out, enc_hidden = model.encoder(src, src_len)

    # local bridge: resize enc_hidden -> dec_hidden depth 
    dec_layers = model.decoder.rnn.num_layers
    def resize(t):
        if t.size(0) == dec_layers:
            return t
        elif t.size(0) < dec_layers:                  # repeat last layer
            pad = t[-1:].expand(dec_layers - t.size(0), -1, -1)
            return torch.cat([t, pad], 0)
        else:                                         # slice extra layers
            return t[:dec_layers]

    if isinstance(enc_hidden, tuple):                # LSTM (h,c)
        h, c = enc_hidden
        dec_hidden = (resize(h), resize(c))
    else:                                            # GRU
        dec_hidden = resize(enc_hidden)

    mask = (src != model.pad_idx)
    toks, attn_rows = [], []
    tok = torch.full((1,), tgt_v.sos_idx, device=device)

    for _ in range(max_len):
        out, dec_hidden, attn = model.decoder(tok, dec_hidden, enc_out, mask)
        attn_rows.append(attn.squeeze(0).cpu())      # [T_src]
        tok = out.argmax(1)
        if tok.item() == tgt_v.eos_idx:
            break
        toks.append(tok.item())

    return toks, torch.stack(attn_rows)              # [T_tgt, T_src]

# 3. collect first 9 samples 
import random

# Convert the test loader to a list (this loads everything into memory)
all_test_samples = list(loaders["test"])

# Randomly select 9 unique samples
random_samples = random.sample(all_test_samples, 9)

# Format them like your original loop
samples = [(i, src.to(device), src_len.to(device)) for i, (src, src_len, _) in enumerate(random_samples)]

# 4. plot a 3×3 grid
fig, axes = plt.subplots(3, 3, figsize=(12, 10))

for (idx, src, src_len), ax in zip(samples, axes.flatten()):
    gen_ids, attn = infer_with_attn(src, src_len)      # attn: [T_tgt, T_src]
    attn_np = attn.detach().numpy()

    # decode strings
    src_txt = src_vocab.decode(src[0].cpu(), strip_specials=True)
    tgt_txt = tgt_vocab.decode(gen_ids,           strip_specials=True)

    T_src, T_tgt = len(src_txt), len(tgt_txt)

    # plot heat-map
    ax.imshow(attn_np, aspect='auto', cmap='Blues_r')

    # put chars on axes
    ax.set_xticks(np.arange(T_src))
    ax.set_xticklabels(list(src_txt), fontsize=11)
    ax.set_yticks(np.arange(T_tgt))
    ax.set_yticklabels(list(tgt_txt), fontsize=11)

    ax.xaxis.tick_top()
    plt.setp(ax.get_xticklabels(), ha="center", va="bottom")

    # faint grid
    ax.set_xticks(np.arange(-.5, T_src, 1), minor=True)
    ax.set_yticks(np.arange(-.5, T_tgt, 1), minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)

plt.tight_layout()
plt.savefig("attention_grid.png", dpi=160)
print("Saved attention_grid.png")