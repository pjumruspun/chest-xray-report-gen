from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

UNK_TOKEN = "<unk>"

def cnn_rnn_tokenizer(y_train: np.array) -> Tokenizer:
    """
    Create Tensorflow tokenizer and fit it on y_train
    """
    
    filters='!"#$%&()*+,-/:;=?@[\\]^_`{|}~\t\n'
    tokenizer = Tokenizer(filters=filters, oov_token=UNK_TOKEN)
    tokenizer.fit_on_texts(y_train)
    
    return tokenizer