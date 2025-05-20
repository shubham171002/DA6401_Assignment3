# models.py -- Attention Mechanism

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from data   import get_dataloaders
from models import Encoder, Decoder, Seq2Seq


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hid_size, layers=1, cell='LSTM', dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        rnn_cls = {'LSTM': nn.LSTM, 'GRU': nn.GRU}[cell]
        self.rnn = rnn_cls(emb_size,
                           hid_size,
                           num_layers=layers,
                           dropout=dropout if layers>1 else 0.0,
                           batch_first=True,
                           bidirectional=False)

    def forward(self, src, lengths):
        # src: [B, T], lengths: [B]
        embedded = self.embedding(src)  # [B, T, E]
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, hidden = self.rnn(packed)
        outputs, _ = pad_packed_sequence(packed_out, batch_first=True)  # [B, T, H]
        return outputs, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, enc_hid, dec_hid):
        super().__init__()
        self.attn = nn.Linear(enc_hid + dec_hid, dec_hid)
        self.v = nn.Linear(dec_hid, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask):
        # hidden: [B, H], encoder_outputs: [B, T, H], mask: [B, T]
        B, T, H = encoder_outputs.size()
        hidden = hidden.unsqueeze(1).repeat(1, T, 1)               # [B, T, H]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # [B, T, H]
        scores = self.v(energy).squeeze(2)                        # [B, T]
        scores = scores.masked_fill(~mask, -1e9)
        return torch.softmax(scores, dim=1)                       # [B, T]


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_size, enc_hid, dec_hid, layers=1, cell='LSTM', dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.attention = BahdanauAttention(enc_hid, dec_hid)
        rnn_cls = {'LSTM': nn.LSTM, 'GRU': nn.GRU}[cell]
        self.rnn = rnn_cls(emb_size + enc_hid,
                           dec_hid,
                           num_layers=layers,
                           dropout=dropout if layers>1 else 0.0,
                           batch_first=True)
        self.fc = nn.Linear(dec_hid + enc_hid + emb_size, vocab_size)

    def forward(self, input_token, hidden, encoder_outputs, mask):
        # input_token: [B], hidden: tuple or tensor, encoder_outputs: [B, T, H]
        B = input_token.size(0)
        embedded = self.embedding(input_token).unsqueeze(1)       # [B, 1, E]
        # extract top-layer hidden state for attention
        if isinstance(hidden, tuple):  # LSTM
            dec_h = hidden[0][-1]   # [B, H]
        else:
            dec_h = hidden[-1]      # [B, H]
        attn_weights = self.attention(dec_h, encoder_outputs, mask)  # [B, T]
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # [B,1,H]
        rnn_input = torch.cat((embedded, context), dim=2)           # [B,1,E+H]
        out, hidden = self.rnn(rnn_input, hidden)
        out = out.squeeze(1)                                        # [B, H]
        context = context.squeeze(1)                                # [B, H]
        embedded = embedded.squeeze(1)                              # [B, E]
        pred = self.fc(torch.cat((out, context, embedded), dim=1))  # [B, V]
        return pred, hidden, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, pad_idx, device='cpu'):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.device = device

    def forward(self, src, src_lens, tgt):
        enc_out, hidden = self.encoder(src, src_lens)
        mask = (src != self.pad_idx)
        B, T = tgt.size()
        outputs = torch.zeros(B, T-1, self.decoder.fc.out_features, device=self.device)
        input_tok = tgt[:, 0]  # <sos>
        for t in range(1, T):
            out, hidden, _ = self.decoder(input_tok, hidden, enc_out, mask)
            outputs[:, t-1] = out
            input_tok = tgt[:, t]
        return outputs

    def infer_greedy(self, src, src_lens, tgt_vocab, max_len=50):
        enc_out, hidden = self.encoder(src, src_lens)
        mask = (src != self.pad_idx)
        B = src.size(0)
        input_tok = torch.full((B,), tgt_vocab.sos_idx, device=self.device, dtype=torch.long)
        generated = []
        for _ in range(max_len):
            out, hidden, _ = self.decoder(input_tok, hidden, enc_out, mask)
            input_tok = out.argmax(1)
            generated.append(input_tok.unsqueeze(1))
            if (input_tok == tgt_vocab.eos_idx).all():
                break
        return torch.cat(generated, dim=1)
