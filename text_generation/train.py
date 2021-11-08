import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

from dataset import balance_shuffle, create_dataset, train_val_test_split
from lstm import CNN_Encoder, RNN_Decoder, loss_function
from preprocess import configs
from preprocess import configs as preprocess_configs
from preprocess import load_csv, load_image_mappings
from utils import create_embedding_matrix, get_max_report_len

MAX_LEN = get_max_report_len()

configs = {
    'train_ratio': 0.75,
    'val_ratio': 0.10,
    'test_ratio': 0.15,
    'learning_rate': 1e-3,
    "embedding_dim": 200,
    "decoder_units": 80,
    'epochs': 40,
    'batch_size': 16,
}

image_mappings, tokenizer, encoder, decoder, optimizer = None, None, None, None, None
START_TOK = preprocess_configs['START_TOK']


def files_exist():
    """
    Check if image mappings and report df exists
    """
    pickle_exist = os.path.exists(preprocess_configs['pickle_file_path'])
    csv_exist = os.path.exists(preprocess_configs['csv_file_path'])

    if not pickle_exist:
        print(
            f"Pickle file not found, expected to be at: {preprocess_configs['pickle_file_path']}")

    if not csv_exist:
        print(
            f"Report csv file not found, expected to be at: {preprocess_configs['csv_file_path']}")

    return pickle_exist and csv_exist


def texts_to_sequences(texts) -> np.array:
    seqs = []
    for sentence in texts:
        sent = []
        for word in sentence.split():
            decoded = tokenizer.word_index.get(word.decode('utf-8'))
            if decoded == None:
                decoded = 0
            sent.append(decoded)
        seqs.append(np.asarray(sent))
    return np.asarray(seqs)


def load_features(id_, report):
    global image_mappings
    train_seq = texts_to_sequences([report])
    train_seq = pad_sequences(
        train_seq, MAX_LEN, padding='post', dtype=np.int32)
    img_feature = image_mappings[id_.decode('utf-8')]
    return img_feature, train_seq[0]


@tf.function
def train_step(img_tensor, target, train=True):
    loss = 0
    # initializing the hidden state for each batch
    # because the captions are not related from image to image
    hidden = decoder.reset_state(batch_size=target.shape[0])
    dec_input = tf.expand_dims(
        [tokenizer.word_index[START_TOK]] * target.shape[0], 1)

    with tf.GradientTape() as tape:
        features = encoder(img_tensor)

        for i in range(1, target.shape[1]):  # Broken here
            # passing the features through the decoder
            predictions, hidden, _ = decoder(dec_input, features, hidden)

            calculated_loss = loss_function(target[:, i], predictions)
            loss += calculated_loss
            # using teacher forcing
            dec_input = tf.expand_dims(target[:, i], 1)

    total_loss = (loss / int(target.shape[1]))

    trainable_variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, trainable_variables)

    if train:
        optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, total_loss


def train(train_generator, val_generator, train_size, val_size):
    train_losses = []
    val_losses = []

    best_loss = np.inf

    # Train
    for epoch in range(configs['epochs']):
        print(f"Epoch {epoch+1}")
        start = time.time()
        total_train_loss = 0
        total_val_loss = 0

        print("Training...")
        for (batch, (img_tensor, target)) in enumerate(tqdm(train_generator)):
            batch_loss, t_loss = train_step(img_tensor, target, train=True)
            total_train_loss += t_loss

        print("Validating...")
        for (batch, (img_tensor, target)) in enumerate(tqdm(val_generator)):
            batch_loss, t_loss = train_step(img_tensor, target, train=False)
            total_val_loss += t_loss

        train_loss = total_train_loss / train_size
        val_loss = total_val_loss / val_size

        # Log
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Should save model if val loss is the best here
        if val_loss < best_loss:
            print("New best!")
            best_loss = val_loss
            encoder.save_weights(
                f'weights/attention/enc_attention.h5')
            decoder.save_weights(
                f'weights/attention/dec_attention.h5')

        print(f'Train loss: {train_loss:.6f}\tVal loss: {val_loss:.6f}')
        print(f'Time taken for this epoch {time.time()-start:.3f} sec\n')

    plt.plot(train_losses, label='train loss')
    plt.plot(val_losses, label='val loss')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.show()


def main():
    global image_mappings, tokenizer, encoder, decoder, optimizer

    # Check if necessary dataset file exists
    if files_exist():

        image_mappings = load_image_mappings()
        df = load_csv()
    else:
        raise Exception(
            "Please run preprocess.py to create pickle and csv file first.")

    # Balance shuffling
    balanced_df = balance_shuffle(df)

    X = balanced_df['img_path'].values
    Y = balanced_df['report'].values

    x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(
        X, Y, configs['train_ratio'], configs['val_ratio'], configs['test_ratio'])

    print("Data shapes:")
    print(x_train.shape)
    print(x_val.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_val.shape)
    print(y_test.shape)

    # Create embeddings
    print("Creating embedding matrix...")
    tokenizer, embedding_matrix, vocab_size, w2v_size = create_embedding_matrix(
        Y)

    print("Building datasets...")
    # Build dataset
    train_generator = create_dataset(
        x_train, y_train, load_features, batch_size=configs['batch_size'])
    val_generator = create_dataset(
        x_val, y_val, load_features, batch_size=configs['batch_size'])
    test_generator = create_dataset(
        x_test, y_test, load_features, batch_size=configs['batch_size'])

    # Create models
    print("Creating models...")
    encoder = CNN_Encoder(configs['embedding_dim'],
                          batch_size=configs['batch_size'])
    decoder = RNN_Decoder(
        configs['embedding_dim'], configs['decoder_units'], vocab_size, embedding_matrix)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=configs['learning_rate'])

    # Train
    print("Start training with configs:")
    print(configs)
    train(train_generator, val_generator, len(x_train), len(x_val))


if __name__ == '__main__':
    main()
