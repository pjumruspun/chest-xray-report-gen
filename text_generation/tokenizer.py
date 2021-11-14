from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from preprocess import load_csv
from configs import configs

UNK_TOKEN = "<unk>"


def cnn_rnn_tokenizer() -> Tokenizer:
    """
    Create Tensorflow tokenizer and fit it on reports
    Reports are generated from load_csv function
    """

    df = load_csv()
    reports = df['report'].values

    filters = '!"#$%&()*+,-/:;=?@[\\]^_`{|}~\t\n'
    tokenizer = Tokenizer(filters=filters, oov_token=UNK_TOKEN)
    tokenizer.fit_on_texts(reports)

    return tokenizer

def decode_report(tokenizer, arr):
    arr = np.asarray(arr)
    res = [[] for _ in range(arr.shape[0])]

    for i in range(arr.shape[0]):
        if arr.ndim == 1:
            for j in range(len(arr[i])):
                if arr[i][j] != 0:
                    word = tokenizer.index_word[arr[i][j]]
                    if word != configs['START_TOK'] and word != configs['STOP_TOK']:
                        res[i].append(word)
        else:
            for j in range(arr.shape[1]):
                if arr[i][j] != 0:
                    word = tokenizer.index_word[arr[i][j]]
                    if word != configs['START_TOK'] and word != configs['STOP_TOK']:
                        res[i].append(word)
        res[i] = ' '.join(res[i])

    return res