import os
import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn.model_selection import train_test_split
from utils import SEED, get_max_report_len, create_embedding_matrix
from preprocess import load_csv, load_image_mappings
from preprocess import configs as preprocess_configs


from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = get_max_report_len()

configs = {
    'batch_size': 1,
    'train_ratio': 0.75,
    'val_ratio': 0.10,
    'test_ratio': 0.15,
}

def train_val_test_split(X, Y, train, val, test):
    # Fixed random seed
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=test, random_state=SEED)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=val*(1/(1-test)), random_state=SEED)
    return x_train, x_val, x_test, y_train, y_val, y_test


def balance_shuffle(cleaned_df) -> pd.DataFrame:
    """
    Assuming reports with finding exist more than reports with no finding
    """

    df = cleaned_df.copy()
    no_finding = df[df['Problems'] == 'normal'].reset_index(drop=True)
    finding = df[df['Problems'] != 'normal'].reset_index(drop=True)

    f_count = finding.shape[0]
    nf_count = no_finding.shape[0]

    f_processed = 0
    nf_processed = 0

    results = []

    while f_processed < f_count and nf_processed < nf_count:
        # Calculate how many data with no finding to pick for this iteration
        f_left = f_count - f_processed
        nf_left = nf_count - nf_processed
        f_ratio = f_left // nf_left  # 2

        # Calculate the index to choose
        nf_idx = nf_processed
        f_start = f_processed
        f_end = f_processed + f_ratio

        results.append(no_finding.iloc[[nf_idx]])
        results.append(finding.iloc[f_start:f_end, :])

        f_processed += f_ratio
        nf_processed += 1

    print(f"With finding processed: {f_processed}")
    print(f"No finding processed: {nf_processed}")

    merged = pd.concat(results)
    return merged.reset_index(drop=True)

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


def create_dataset(imgpaths, reports, load_features, batch_size=16):
    # dataset = tf.data.Dataset.from_tensor_slices((balanced_df['imgpath'].values, balanced_df['report'].values, balanced_df['No Finding'].values))

    dataset = tf.data.Dataset.from_tensor_slices((imgpaths, reports))
    dataset = dataset.map(lambda item1, item2: tf.numpy_function(load_features, [item1, item2], [tf.float32, tf.int32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.shuffle(buffer_size=len(imgpaths), seed=SEED)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset

def get_train_materials():
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

    return train_generator, val_generator, test_generator, tokenizer, embedding_matrix, vocab_size, len(x_train), len(x_val), len(x_test)