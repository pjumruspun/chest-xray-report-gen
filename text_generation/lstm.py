import tensorflow as tf
from datetime import datetime

DEFAULT_WEIGHT_DIR = 'weights/attention/'


def timestamp():
    d = str(datetime.now())
    d = d.replace(' ', '_')
    d = d.replace(':', '-')
    d = d.replace('.', '-')
    return d


def loss_function(real, pred):
    """
    Masked sparse categorical crossentropy loss
    """
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)
        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)

        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # attention_hidden_layer shape == (batch_size, 64, units)
        attention_hidden_layer = (tf.nn.tanh(self.W1(features) +
                                             self.W2(hidden_with_time_axis)))

        # score shape == (batch_size, 64, 1)
        # This gives you an unnormalized score for each image feature.
        score = self.V(attention_hidden_layer)

        # attention_weights shape == (batch_size, 64, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class CNN_Encoder(tf.keras.Model):
    """
    Source: https://www.tensorflow.org/tutorials/text/image_captioning

    Since you have already extracted the features and dumped it
    This encoder passes those features through a Fully connected layer
    """

    def __init__(self, embedding_dim, batch_size):
        super().__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.reshape = tf.keras.layers.Reshape(
            (64, embedding_dim), input_shape=(batch_size, 8, 8, embedding_dim))
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        x = self.reshape(x)
        return x

    def save_model(self, directory=DEFAULT_WEIGHT_DIR):
        print("Saving weights...")
        self.save_weights(directory + 'encoder_' + timestamp() + '.h5')

    def load_model(self, directory=DEFAULT_WEIGHT_DIR):
        print("Loading weights...")
        self.save_weights(directory + 'encoder.h5')


class RNN_Decoder(tf.keras.Model):
    """
    Source: https://www.tensorflow.org/tutorials/text/image_captioning
    """

    def __init__(self, embedding_dim, units, embedding_matrix):
        super().__init__()
        vocab_size = embedding_matrix.shape[0]
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                                   mask_zero=True, trainable=True,
                                                   weights=[embedding_matrix])
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        # defining attention as a separate model
        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
        context_vector, attention_weights = self.attention(features, hidden)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)

        x = self.fc1(output)
        x = tf.reshape(x, (-1, x.shape[2]))
        x = self.fc2(x)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))

    def save_model(self, directory=DEFAULT_WEIGHT_DIR):
        print("Saving weights...")
        self.save_weights(directory + 'decoder_' + timestamp() + '.h5')

    def load_model(self, directory=DEFAULT_WEIGHT_DIR):
        print("Loading weights...")
        self.load_weights(directory + 'decoder.h5')
