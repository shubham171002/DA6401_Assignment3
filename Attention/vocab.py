# vocab.py

import json

class CharVocab:
    def __init__(self, tokens=None, specials=['<pad>','<sos>','<eos>','<unk>']):
        self.specials = specials
        self.idx2char = list(specials) + (tokens or [])
        self.char2idx = {ch:i for i,ch in enumerate(self.idx2char)}

    @classmethod
    def build_from_texts(cls, texts):
        chars = sorted({c for line in texts for c in line})
        return cls(tokens=chars)

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.idx2char, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        idx2char = json.load(open(path, encoding='utf-8'))
        inst = cls(tokens=[])
        inst.idx2char = idx2char
        inst.char2idx = {c:i for i,c in enumerate(idx2char)}
        return inst

    def encode(self, text, add_sos=False, add_eos=False):
        seq = []
        if add_sos: seq.append(self.char2idx['<sos>'])
        for c in text:
            seq.append(self.char2idx.get(c, self.char2idx['<unk>']))
        if add_eos: seq.append(self.char2idx['<eos>'])
        return seq

    def decode(self, idxs, strip_specials=True):
        chars = [self.idx2char[i] for i in idxs]
        if strip_specials:
            chars = [c for c in chars if c not in self.specials]
        return ''.join(chars)

    @property
    def pad_idx(self): return self.char2idx['<pad>']
    @property
    def sos_idx(self): return self.char2idx['<sos>']
    @property
    def eos_idx(self): return self.char2idx['<eos>']
    @property
    def unk_idx(self): return self.char2idx['<unk>']
    @property
    def size(self): return len(self.idx2char)
