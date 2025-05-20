# models.py  -- No Attention Seq2Seq model 
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from data   import get_dataloaders
from models import Encoder, Decoder, Seq2Seq


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hid_size,
                 layers=1, cell='LSTM', dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        rnn_cls = {'LSTM': nn.LSTM, 'GRU': nn.GRU, 'RNN': nn.RNN}[cell]
        self.rnn = rnn_cls(
            emb_size, hid_size, num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True, bidirectional=False
        )

    def forward(self, src, lengths):
        emb = self.embedding(src)                                   # [B,T,E]
        packed = pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        enc_out, hidden = self.rnn(packed)               
        return None, hidden                                       


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_size, dec_hid,
                 layers=1, cell='LSTM', dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        rnn_cls = {'LSTM': nn.LSTM, 'GRU': nn.GRU, 'RNN': nn.RNN}[cell]
        self.rnn = rnn_cls(
            emb_size, dec_hid, num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True
        )
        self.fc = nn.Linear(dec_hid, vocab_size)

    def forward(self, input_tok, hidden):
        # input_tok: [B]  hidden: (layers,B,H) or tensor
        emb = self.embedding(input_tok).unsqueeze(1)                # [B,1,E]
        out, hidden = self.rnn(emb, hidden)                         # [B,1,H]
        pred = self.fc(out.squeeze(1))                              # [B,V]
        return pred, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, pad_idx, device='cpu'):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.device  = device

    # bridge enc→dec hidden if layer counts differ
    def _bridge(self, enc_hidden):
        dec_layers = self.decoder.rnn.num_layers

        def resize(t):
            if t.size(0) == dec_layers:
                return t
            elif t.size(0) < dec_layers:
                pad = t[-1:].expand(dec_layers - t.size(0), -1, -1)
                return torch.cat([t, pad], 0)
            else:
                return t[:dec_layers]

        if isinstance(enc_hidden, tuple):            # LSTM
            h, c = enc_hidden
            return resize(h), resize(c)
        return resize(enc_hidden)                    # GRU / RNN

    def forward(self, src, src_lens, tgt):
        _, enc_hidden = self.encoder(src, src_lens)
        hidden = self._bridge(enc_hidden)

        B, T = tgt.size()
        outputs = torch.zeros(
            B, T - 1, self.decoder.fc.out_features, device=self.device
        )
        input_tok = tgt[:, 0]                   
        for t in range(1, T):
            out, hidden = self.decoder(input_tok, hidden)
            outputs[:, t - 1] = out
            input_tok = tgt[:, t]                    # teacher forcing
        return outputs

    def infer_greedy(self, src, src_lens, tgt_vocab, max_len=50):
        _, enc_hidden = self.encoder(src, src_lens)
        hidden = self._bridge(enc_hidden)

        B = src.size(0)
        input_tok = torch.full(
            (B,), tgt_vocab.sos_idx, dtype=torch.long, device=self.device
        )
        gen = []
        for _ in range(max_len):
            out, hidden = self.decoder(input_tok, hidden)
            input_tok = out.argmax(1)
            gen.append(input_tok.unsqueeze(1))
            if (input_tok == tgt_vocab.eos_idx).all():
                break
        return torch.cat(gen, 1)