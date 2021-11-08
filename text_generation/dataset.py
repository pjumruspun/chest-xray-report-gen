import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn.model_selection import train_test_split
from utils import SEED


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


def create_dataset(imgpaths, reports, load_features, batch_size=16):
    # dataset = tf.data.Dataset.from_tensor_slices((balanced_df['imgpath'].values, balanced_df['report'].values, balanced_df['No Finding'].values))

    dataset = tf.data.Dataset.from_tensor_slices((imgpaths, reports))
    dataset = dataset.map(lambda item1, item2: tf.numpy_function(load_features, [item1, item2], [tf.float32, tf.int32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.shuffle(buffer_size=len(imgpaths), seed=SEED)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset