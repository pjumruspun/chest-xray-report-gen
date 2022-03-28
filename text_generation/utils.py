import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms
from gensim.models import KeyedVectors
from preprocess import load_csv
from configs import configs
import os

SEED = 0

CONDITIONS = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
              'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
              'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
              'Support Devices', 'No Finding']

normalize = transforms.Normalize(
    mean=[0.4684, 0.4684, 0.4684], 
    std=[0.3021, 0.3021, 0.3021]
)

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    normalize
])

evaluate_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize
])



def create_embedding_matrix(tokenizer, use_cache=True) -> np.array:
    """
    Create embedding matrix with size of (vocab_size, w2v_size)
    In this case, w2v_size is always 200 using pubmed2018 pretrained embeddings
    """

    file_exist = os.path.exists(configs['emb_matrix_path'])

    if use_cache and file_exist:
        print(f"Using cached embedding matrix at {configs['emb_matrix_path']} ...")
        embedding_matrix = np.load(configs['emb_matrix_path'])
        vocab_size = embedding_matrix.shape[0]
        w2v_size = embedding_matrix.shape[1]
    else:
        # Create a new embedding matrix and save
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
        
        # Save here
        np.save(configs['emb_matrix_path'], embedding_matrix)

    return embedding_matrix, vocab_size, w2v_size


def get_max_report_len() -> int:
    df = load_csv()
    lengths = df['report'].apply(lambda x: x.split()).str.len()
    return lengths.max()

def decode_sequences(tokenizer, seqs):
    """
    Parameters:
        tokenizer: torch.data.utils tokenizer
        seqs: list of lists or alike structure containing indices
    Returns:
        list of sentences (list of str)
    """

    exempt_words = ['<pad>', '<startseq>', '<endseq>', '<unk>']
    exempt_toks = [tokenizer.stoi[word] for word in exempt_words]
    sentences = [' '.join([tokenizer.itos[int(idx)] for idx in seq if idx not in exempt_toks]) for seq in seqs]
    return sentences
