from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, Vocab
from collections import Counter
import pandas as pd
import torch
from configs import configs

class TokenizerWrapper():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.stoi = self.tokenizer.stoi
        self.itos = self.tokenizer.itos
        self.lookup_indices = self.tokenizer.lookup_indices
    
    def decode(self, seqs, exempt_words=['<pad>', '<startseq>', '<endseq>', '<unk>']):
        exempt_toks = [self.tokenizer.stoi[word] for word in exempt_words]
        sentences = [' '.join([self.tokenizer.itos[int(idx)] for idx in seq if idx not in exempt_toks]) for seq in seqs]
        return sentences

    # def encode(self, word_seqs, pad='max', device=torch.device('cpu')):
    #     """
    #     Parameters:
    #         word_seqs (numpy.ndarray): Input array
    #         pad (str('max') or int): 'max' for padding to the longest length in the array, else pad to length = this parameter

    #     """
    #     if not (isinstance(word_seqs, np.ndarray) or isinstance(word_seqs, list)):
    #         raise TypeError(f"Unknown input type: {type(word_seqs)}")

    #     dim = len(word_seqs.shape)

    #     if dim == 1:
    #         encoded = torch.tensor([self.tokenizer.stoi[e] for e in word_seqs], device=device)
    #         return encoded
    #     elif dim == 2:
    #         if pad == 'max':
    #             lengths = [len(e) for e in word_seqs]
    #             length_to_pad = max(lengths)
    #         else:
    #             length_to_pad = pad

    #         # TODO: Pad to length

    #         encoded = torch.tensor([[self.tokenizer.stoi[e] for e in seq] for seq in word_seqs], device=device)
            
    #         return encoded
    #     else:
    #         raise ValueError(f"Invalid dimension: expected dim < 3 but received {dim}")


def create_tokenizer():

    df = pd.read_csv(configs['csv_file_path'])
    words = [word for sent in df['report'].values for word in sent.split()]
    words = Counter(words)
    word_iter = words.keys()
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, word_iter))

    # print(vocab.lookup_indices(['chest', 'normal', 'xxxx', '<startseq>']))
    # print(vocab.lookup_indices(['<startseq>', 'xxxx', 'normal', 'chest']))
    # print(vocab.lookup_indices(['<endseq>', '.']))
    # print(vocab.itos[1])
    # print(vocab.itos[0])
    
    return TokenizerWrapper(vocab)



if __name__ == "__main__":
    import numpy as np
    import torch
    tokenizer = create_tokenizer()
    print(type(tokenizer))
    print(tokenizer.itos[1])
    
    s1 = np.array(['chest', 'normal', 'xxxx', '<startseq>'])
    s2 = np.array([['<startseq>', 'xxxx', 'normal'], ['xxxx', 'normal', 'chest', '<endseq>', '.']])
    # s3 = torch.tensor(['chest', 'normal', 'xxxx', '<startseq>'])
    # s4 = torch.tensor([['<startseq>', 'xxxx', 'normal'], ['xxxx', 'normal', 'chest', '<endseq>', '.']])
    print(tokenizer.encode(s1))
    print(tokenizer.encode(s2, device=torch.device('cuda')))
    # print(tokenizer.encode(s3))
    # print(tokenizer.encode(s4))

    s1 = np.array([1, 2, 3, 4, 5])
    s2 = np.array([[1, 2, 3], [4, 5, 6]])
    s3 = torch.tensor([1, 2, 3, 4, 5, 6])
    s4 = torch.tensor([[1, 2, 3], [4, 5, 6]])
