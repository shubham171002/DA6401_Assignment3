# Question 6 - Visualization Connectivity
import sys
import pathlib
path = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(path))

import matplotlib as mpl

font_path = "Noto_Sans_Gujarati/static/NotoSansGujarati-Regular.ttf"

# Register the font with matplotlib
mpl.font_manager.fontManager.addfont(font_path)
mpl.rcParams["font.family"] = "Noto Sans Gujarati"


import torch, numpy as np, matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

from data   import get_dataloaders  
from models import Encoder, Decoder, Seq2Seq

from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import imageio, tempfile, pathlib


BEST_CONFIG = dict(
    emb_size      = 128,
    hidden_size   = 128,
    layers        = 2,
    cell          = "GRU",
    dropout       = 0.5,
    lr            = 1e-4,
    batch_size    = 32,
    epochs        = 20,
)


BEST_CKPT = "prediction_attention/attn_model.pth"
LANG      = "gu"
BATCH     = 1

device = "cuda" if torch.cuda.is_available() else "cpu"
loaders, src_vocab, tgt_vocab = get_dataloaders(LANG, batch_size=BATCH, device=device)

# rebuild best model
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


# 2. helper: run model & return prediction + attention matrix
def infer_with_attn(src, src_len, max_len=50):
    """Returns (generated token ids, attention matrix)"""

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
    tok = torch.full((1,), tgt_vocab.sos_idx, device=device)

    for _ in range(max_len):
        out, dec_hidden, attn = model.decoder(tok, dec_hidden, enc_out, mask)
        attn_rows.append(attn.squeeze(0).cpu())      # [T_src]
        tok = out.argmax(1)
        if tok.item() == tgt_vocab.eos_idx:
            break
        toks.append(tok.item())

    return toks, torch.stack(attn_rows)              # [T_tgt, T_src]

# 3. helper: nice curved line between (x0,y0) and (x1,y1)
def bezier(x0, y0, x1, y1, bend=0.25):
    """Return Path object for a quadratic Bézier curve."""
    ctrl_x, ctrl_y = (x0 + x1) / 2, y0 + bend
    verts = [(x0, y0), (ctrl_x, ctrl_y), (x1, y1)]
    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
    return Path(verts, codes)


# 4. side_connectivity.py  –  animated connectivity plot

# assumes: model, infer_with_attn, src_vocab, tgt_vocab, device, loaders already in memory

def make_side_gif(src_chars, tgt_chars, attn_mat, out_gif="side.gif",
                  fade=5, fps=3):
    """
    src_chars : list of str (encoder)
    tgt_chars : list of str (decoder prediction)
    attn_mat  : numpy [T_tgt, T_src]  (rows sum to 1)
    fade      : number of previous frames to keep before link disappears
    Produces a GIF with inputs on the LEFT column, outputs on the RIGHT.
    """
    T_src, T_tgt = len(src_chars), len(tgt_chars)
    y_src = np.linspace(0.95, 0.05, T_src)
    y_tgt = np.linspace(0.95, 0.05, T_tgt)
    x_src, x_tgt = 0.1, 0.9

    # keep frames as numpy arrays -> write once with imageio
    frames = []
    colours = plt.get_cmap("autumn")  # bright orange palette

    for t in range(T_tgt):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

        # draw left & right tokens
        for i, ch in enumerate(src_chars):
            ax.text(x_src-0.03, y_src[i], ch,
                    ha="right", va="center",
                    fontsize=14,
                    color=("tab:red" if i == attn_mat[t].argmax() else "black"))

        for i, ch in enumerate(tgt_chars):
            ax.text(x_tgt+0.03, y_tgt[i], ch,
                    ha="left", va="center",
                    fontsize=14,
                    color=("tab:red" if i == t            else "black"))

        # draw attention links for a sliding window [t-fade … t]
        segs, widths, cols = [], [], []
        for past in range(max(0, t-fade), t+1):
            row = attn_mat[past]
            # top-2 sources for this decoder step
            top_idx = row.argsort()[-2:][::-1]
            for rank, j in enumerate(top_idx):
                if j >= T_src:
                    continue  # skip if attention is to padding or out-of-range
                w = row[j]
                if w < 0.15:         # ignore weak edges
                    continue
                segs.append([(x_src, y_src[j]), (x_tgt, y_tgt[past])])
                widths.append(1.5 + 3*w)
                age    = t - past 
                alpha  = 1.0 - age/(fade+0.5)
                cols.append(colours(0.2 + 0.6*w)[:3] + (alpha,))

        
        ax.add_collection(LineCollection(segs, linewidths=widths,
                                         colors=cols, capstyle="round"))

        # halo on current nodes
        highlight_idx = attn_mat[t].argmax()
        if highlight_idx < len(y_src):
            ax.scatter([x_src], [y_src[highlight_idx]],
                       s=300, edgecolors="orange", facecolors="none", lw=2, alpha=0.6)

        if t < len(y_tgt):
            ax.scatter([x_tgt], [y_tgt[t]],
                       s=300, edgecolors="orange", facecolors="none", lw=2, alpha=0.6)

        # to numpy buffer
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        plt.close(fig)

    imageio.mimsave(out_gif, frames, fps=fps, loop=0)
    print(f"animated connectivity saved → {out_gif}")

# loop over test set, pick 5 random samples
import random, pathlib
random.seed(28)
test_rows = list(loaders["test"])
chosen    = random.sample(range(len(test_rows)), 5)

out_dir = pathlib.Path("side_connectivity_gifs"); out_dir.mkdir(exist_ok=True)

for idx in chosen:
    src, src_len, _ = test_rows[idx]
    src, src_len = src.to(device), src_len.to(device)
    dec_ids, attn = infer_with_attn(src, src_len)

    # tokens -> glyph lists
    src_ids = src[0, :src_len.item()].cpu().tolist()
    if src_ids and src_ids[0] == src_vocab.sos_idx:  src_ids = src_ids[1:]
    if src_ids and src_ids[-1] == src_vocab.eos_idx: src_ids = src_ids[:-1]
    src_chars = list(src_vocab.decode(src_ids, strip_specials=False))
    tgt_chars = list(tgt_vocab.decode(dec_ids, strip_specials=True))

    gif_name = out_dir / f"side_connectivity_{idx}.gif"
    make_side_gif(src_chars, tgt_chars, attn.detach().numpy(),
                  out_gif=str(gif_name), fps=3)