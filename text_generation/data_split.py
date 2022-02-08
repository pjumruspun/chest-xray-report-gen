import os
import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn.model_selection import train_test_split
from utils import SEED, get_max_report_len, create_embedding_matrix
from preprocess import load_csv, load_image_mappings
from tokenizer import cnn_rnn_tokenizer


from tensorflow.keras.preprocessing.sequence import pad_sequences
from configs import configs

MAX_LEN = get_max_report_len()


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
    pickle_exist = os.path.exists(configs['pickle_file_path'])
    csv_exist = os.path.exists(configs['csv_file_path'])

    if not pickle_exist:
        print(
            f"Pickle file not found, expected to be at: {configs['pickle_file_path']}")

    if not csv_exist:
        print(
            f"Report csv file not found, expected to be at: {configs['csv_file_path']}")

    return pickle_exist and csv_exist


def create_dataset(imgpaths, reports, load_features, batch_size=16):
    dataset = tf.data.Dataset.from_tensor_slices((imgpaths, reports))
    dataset = dataset.map(lambda item1, item2: tf.numpy_function(load_features, [item1, item2], [tf.float32, tf.int32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.shuffle(buffer_size=len(imgpaths), seed=SEED)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def get_train_materials(test_batch_size=None):
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

    # For saving df
    train_df, val_df, test_df, _, _, _ = train_val_test_split(
        balanced_df, Y, configs['train_ratio'], configs['val_ratio'], configs['test_ratio'])

    train_df.to_csv(configs['train_csv'], index=False)
    val_df.to_csv(configs['val_csv'], index=False)
    test_df.to_csv(configs['test_csv'], index=False)

    print("Data shapes:")
    print(x_train.shape)
    print(x_val.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_val.shape)
    print(y_test.shape)

    # Get tokenizer
    tokenizer = cnn_rnn_tokenizer()

    # Create embeddings
    print("Creating embedding matrix...")
    embedding_matrix, vocab_size, _ = create_embedding_matrix(tokenizer)

    print("Building datasets...")
    # Build dataset
    train_generator = create_dataset(
        x_train, y_train, load_features, batch_size=configs['batch_size'])
    val_generator = create_dataset(
        x_val, y_val, load_features, batch_size=configs['batch_size'])
    if test_batch_size is None:
        test_generator = create_dataset(
            x_test, y_test, load_features, batch_size=configs['test_batch_size'])
    else:
        test_generator = create_dataset(
            x_test, y_test, load_features, batch_size=test_batch_size)

    generators = (train_generator, val_generator, test_generator)
    data = x_train, y_train, x_val, y_val, x_test, y_test

    return generators, data, tokenizer, embedding_matrix, vocab_size


def export_data_csv() -> None:
    """
    Load all dataframe and shuffle them, then split them into train val test
    Save all of those dataframes into csv according to configs
    """
    df = load_csv()

    # Balance shuffling
    balanced_df = balance_shuffle(df)

    Y = balanced_df['report'].values

    # For saving df
    train_df, val_df, test_df, _, _, _ = train_val_test_split(
        balanced_df, Y, configs['train_ratio'], configs['val_ratio'], configs['test_ratio'])

    train_df.to_csv(configs['train_csv'], index=False)
    val_df.to_csv(configs['val_csv'], index=False)
    test_df.to_csv(configs['test_csv'], index=False)


if __name__ == '__main__':
    get_train_materials()
