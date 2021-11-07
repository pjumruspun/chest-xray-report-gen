
import keras
import os
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

OUTPUT_LAYER_SIZE = 14
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'weights/brucechou1983_CheXNet_Keras_0.3.0_weights.h5')

class ChexNet(keras.Model):
    def __init__(self, input_shape=(224, 224, 3), pooling=None):
        """
        Input shape: (224, 224, 3)
        Output shape: (7, 7, 1024)
        Output shape with pooling='avg': (1024,)
        """

        super(ChexNet, self).__init__()

        # Instantiate DenseNet121 with no weight
        densenet = DenseNet121(include_top=False, weights=None, input_shape=input_shape, pooling=pooling)

        # Create a new output layer for the empty densenet
        output = Dense(OUTPUT_LAYER_SIZE, activation="sigmoid", name="predictions")(densenet.output)

        # Build model and load weights
        chex_model = Model(inputs=densenet.input, outputs=output)
        chex_model.load_weights(MODEL_DIR)

        # Real model uses second last layer of Chexnet as an output
        self.feature_model = Model(inputs=chex_model.input, outputs=chex_model.layers[-2].output)

    def call(self, images):
        return self.feature_model.predict(images)

