import numpy as np
from tqdm import tqdm
from gensim.models import KeyedVectors
from preprocess import load_csv
from configs import configs

SEED = 0


def create_embedding_matrix(tokenizer) -> np.array:
    """
    Create embedding matrix with size of (vocab_size, w2v_size)
    In this case, w2v_size is always 200 using pubmed2018 pretrained embeddings
    """

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

    return embedding_matrix, vocab_size, w2v_size


def get_max_report_len() -> int:
    df = load_csv()
    lengths = df['report'].apply(lambda x: x.split()).str.len()
    return lengths.max()
