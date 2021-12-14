from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from preprocess import load_csv
from collections import Counter
from torch.nn.utils.rnn import pad_sequence

def create_tokenizer():

    df = load_csv()
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
    
    return vocab



if __name__ == "__main__":
    create_tokenizer()