from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from preprocess import load_csv

UNK_TOKEN = "<unk>"

def cnn_rnn_tokenizer() -> Tokenizer:
    """
    Create Tensorflow tokenizer and fit it on reports
    Reports are generated from load_csv function
    """
    
    df = load_csv()
    reports = df['report'].values
    
    filters='!"#$%&()*+,-/:;=?@[\\]^_`{|}~\t\n'
    tokenizer = Tokenizer(filters=filters, oov_token=UNK_TOKEN)
    tokenizer.fit_on_texts(reports)
    
    return tokenizer