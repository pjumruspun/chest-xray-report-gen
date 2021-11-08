import numpy as np
import os
from tqdm import tqdm
from gensim.models import KeyedVectors
from preprocess import load_csv
from tokenizer import cnn_rnn_tokenizer

SEED = 0

configs = {
    'pretrained_emb_path': os.path.join(os.path.dirname(__file__), 'weights/pubmed2018_w2v_200D/pubmed2018_w2v_200D.bin'),
}

def create_embedding_matrix(reports):
    # Config tokenizer
    tokenizer = cnn_rnn_tokenizer(reports)

    # Get embedding matrix path from configs
    emb_matrix_path = configs['pretrained_emb_path']

    vocab_size = len(tokenizer.word_index.keys()) + 1

    # Load word2vec
    word_vectors = KeyedVectors.load_word2vec_format(
        emb_matrix_path, binary=True)
    w2v_size = word_vectors.vector_size

    # Create embedding matrix
    embedding_matrix = np.zeros((vocab_size, w2v_size))
    for word, i in tqdm(tokenizer.word_index.items()):
        if word in word_vectors.key_to_index:
            embedding_matrix[i] = word_vectors.get_vector(word)
        else:
            continue

    return tokenizer, embedding_matrix, vocab_size, w2v_size


def get_max_report_len() -> int:
    df = load_csv()
    lengths = df['report'].apply(lambda x: x.split()).str.len()
    return lengths.max()