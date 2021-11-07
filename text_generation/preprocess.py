import os
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from chexnet import ChexNet
from tqdm import tqdm
from skimage.transform import resize
from skimage.io import imread
from tensorflow.keras.applications.densenet import preprocess_input
from preprocess_text import text_preprocessing, remove_empty

configs = {
    'seed_value': 0,
    'image_path': os.path.join(os.path.dirname(__file__), 'data/images/images_normalized/'),
    'report_csv': os.path.join(os.path.dirname(__file__), 'data/indiana_reports.csv'),
    'projection_csv': os.path.join(os.path.dirname(__file__), 'data/indiana_projections.csv'),
    'pickle_file_path': os.path.join(os.path.dirname(__file__), 'image_features.pickle'),
    'input_shape': (256, 256, 3),
    'START_TOK': '<startseq>',
    'STOP_TOK': '<endseq>',
}


def add_start_stop(text):
    return configs['START_TOK'] + ' ' + text.str.strip() + ' ' + configs['STOP_TOK']


def preprocess_text(df) -> pd.DataFrame:
    """
    Preprocess 'report' column of df according to preprocess_text.py
    """

    cleaned_df = df.copy()
    cleaned_df['report'] = text_preprocessing(df['report'])

    # Remove empty
    cleaned_df = remove_empty(cleaned_df)
    cleaned_df['report'] = add_start_stop(cleaned_df['report'])

    # No manipulation here, just see what's the min and max length
    lengths = cleaned_df['report'].apply(lambda x: x.split()).str.len()
    print(
        f"Max report length = {lengths.max()}, min report length = {lengths.min()}")

    return cleaned_df


def preprocess_images(df) -> dict:
    """
    Preprocess images in df['img_path']
    """

    img_paths = df['img_path'].values

    print(f"Loading {len(img_paths)} images...")
    images = load_images(img_paths)
    print(f"{images.shape=}")

    print(f"Generating image features..")
    image_features = generate_image_features(images)
    print(f"{image_features.shape=}")

    image_mappings = create_image_mappings(img_paths, image_features)
    return image_mappings

def save_image_mappings(image_mappings):
    with open(configs['pickle_file_path'], 'wb') as f:
        print(f"Saving image mappings... ({len(image_mappings)} total images)")
        pickle.dump(image_mappings, f)

def load_image_mappings():
    """
    Function for loading image mappings, for other python file to call
    """

    with open(configs['pickle_file_path'], 'rb') as f:
        image_mappings = pickle.load(f)
        return image_mappings


def load_dataframe() -> pd.DataFrame:
    """
    Returns a joined dataframe consists of uid, filename, projection, and report.
    Using indiana_projections.csv and indiana_reports.csv
    """

    COLUMNS_TO_RETURN = ['uid', 'img_path', 'Problems', 'projection', 'report']

    # Read the csv files
    report_df = pd.read_csv(configs['report_csv'])
    proj_df = pd.read_csv(configs['projection_csv'])

    df = pd.merge(proj_df, report_df, left_on='uid', right_on='uid')
    df['report'] = df['findings'].fillna('') + df['impression'].fillna('')

    # Full path
    df['img_path'] = configs['image_path'] + df['filename']

    return df[COLUMNS_TO_RETURN]


def load_images(img_paths, preprocess=True) -> np.array:
    """
    Load images into np.array by given img_paths
    Also automatically resized to configs['input_shape']
    """

    images = []
    for img_path in tqdm(img_paths):
        image = imread(img_path)
        image = np.asarray(image)
        image = resize(image, configs['input_shape'])
        if preprocess:
            image = preprocess_input(image)
        images.append(image)

    return np.asarray(images)


def generate_image_features(images, batch_size=32) -> np.array:
    """
    Generate image features from images using pretrained ChexNet
    """

    chexnet = ChexNet(input_shape=configs['input_shape'], pooling=None)
    image_features = []

    # batch
    image_generator = tf.data.Dataset.from_tensor_slices(images)
    image_generator = image_generator.batch(batch_size)

    for batch in tqdm(image_generator):
        features_batch = chexnet(batch)
        image_features.extend(features_batch)

    return np.asarray(image_features)


def create_image_mappings(img_paths, image_features) -> dict:

    if len(img_paths) != image_features.shape[0]:
        raise Exception(f"{len(img_paths)} != {len(image_features.shape[0])}")

    return dict(zip(img_paths, image_features))


if __name__ == "__main__":
    df = load_dataframe()

    # preprocess reports
    df = preprocess_text(df)

    # preprocess images
    image_mappings = preprocess_images(df.head(300))

    # save to pickle file
    save_image_mappings(image_mappings)
