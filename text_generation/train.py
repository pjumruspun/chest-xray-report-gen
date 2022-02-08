import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

from data_split import get_train_materials
from model import CNN_Encoder, RNN_Decoder, loss_function
from configs import configs

image_mappings, tokenizer, encoder, decoder, optimizer = None, None, None, None, None
START_TOK = configs['START_TOK']


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
            encoder.save_model()
            decoder.save_model()

        print(f'Train loss: {train_loss:.6f}\tVal loss: {val_loss:.6f}')
        print(f'Time taken for this epoch {time.time()-start:.3f} sec\n')

    plt.plot(train_losses, label='train loss')
    plt.plot(val_losses, label='val loss')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.show()


def main():
    global tokenizer, encoder, decoder, optimizer
    print("Start training with configs:")
    print(configs)

    generators, data, tokenizer, embedding_matrix, _ = get_train_materials()
    train_generator, val_generator, _ = generators
    train_size = data[0].shape[0]
    val_size = data[2].shape[0]

    # Create models
    print("Creating models...")
    encoder = CNN_Encoder(configs['embedding_dim'],
                          batch_size=configs['batch_size'])
    decoder = RNN_Decoder(
        configs['embedding_dim'], configs['decoder_units'], embedding_matrix)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=configs['learning_rate'])

    # Train
    train(train_generator, val_generator, train_size, val_size)


if __name__ == '__main__':
    main()
