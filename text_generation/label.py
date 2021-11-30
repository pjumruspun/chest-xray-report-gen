import numpy as np
import tensorflow as tf

from model import CNN_Encoder, RNN_Decoder
from train import configs as train_configs
from preprocess import configs as preprocess_configs
from utils import get_max_report_len, create_embedding_matrix

MAX_LEN = get_max_report_len()
START_TOK = preprocess_configs['START_TOK']
STOP_TOK = preprocess_configs['STOP_TOK']


def generate_sentence(encoder, decoder, tokenizer, image_features, batch_size) -> str:
    """
    Using image features of size (batch_size, 8, 8, 1024) to generate a whole sentence
    """

    hidden = decoder.reset_state(batch_size=batch_size)

    # Shape should be (batch_size, 8, 8, 1024)
    if image_features.ndim == 3:
        image_features = np.asarray([image_features])

    features = encoder(image_features)

    dec_input = tf.expand_dims(
        [tokenizer.word_index[START_TOK]] * batch_size, 1)
    results = [[] for _ in range(batch_size)]
    dones = [False for _ in range(batch_size)]

    for _ in range(MAX_LEN):
        predictions, hidden, _ = decoder(dec_input, features, hidden)

        # sample from log prob
        # predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        predicted_ids = tf.random.categorical(predictions, 1).numpy()

        # append word
        words = [tokenizer.index_word[id[0]] for id in predicted_ids]

        for i in range(batch_size):
            if words[i] == STOP_TOK:
                dones[i] = True

            if not dones[i]:
                results[i].append(words[i])

        dec_input = predicted_ids

    joined_results = []
    for ls in results:
        joined_results.append(' '.join(ls))

    return joined_results


def prettify(raw_text, delimiter=' ') -> str:
    res = []
    for sentence in raw_text.split('.'):
        if len(sentence.strip()) == 0:
            continue
        sentence = sentence.replace('<startseq>', '')
        sentence = sentence.replace('<endseq>', '')
        res.append(sentence.strip().capitalize() + '.')
    return delimiter.join(res)


def prep_models(embedding_matrix='None') -> tuple((CNN_Encoder, RNN_Decoder)):
    """
    Return prepared encoder and decoder
    Pretrained weights is optionally given as it'll be overwritten anyway
    Given pretrained weights will make this function faster
    """

    # If pretrained embeddings wasn't given, generate
    # Using string of None because if using real None,
    # Python will try to check each element of embedding_matrix array
    if embedding_matrix == 'None':
        embedding_matrix, _, _ = create_embedding_matrix

    encoder = CNN_Encoder(train_configs['embedding_dim'],
                          batch_size=train_configs['batch_size'])
    decoder = RNN_Decoder(
        train_configs['embedding_dim'], train_configs['decoder_units'], embedding_matrix)

    # Warm up model by calling them
    features = encoder(np.zeros((1, 8, 8, 1024)))

    hidden = decoder.reset_state(batch_size=1)

    # dec_input = [[1]]
    dec_input = tf.expand_dims(
        tf.convert_to_tensor(np.ones((1), dtype=int)), 1)

    decoder(dec_input, features, hidden)

    # Load weights

    encoder.load_model()
    decoder.load_model()

    return encoder, decoder
