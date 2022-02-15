import pandas as pd
from utils import SEED
from sklearn.model_selection import train_test_split
from configs import configs

def train_val_test_split(X, Y, train, val, test):
    # Fixed random seed
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=test, random_state=SEED)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=val*(1/(1-test)), random_state=SEED)
    return x_train, x_val, x_test, y_train, y_val, y_test

def split_data_iu_xray():
    """
    Split data into train, val, test, from iu-xray dataset
    """

    df = pd.read_csv(configs['iu_xray_csv_file_path'])
    print(f"Total rows: {df.shape[0]}")
    print(df.head())

    Y = df['report'].values

    train_df, val_df, test_df, _, _, _ = train_val_test_split(
        df, Y, configs['train_ratio'], configs['val_ratio'], configs['test_ratio'])

    # Write CSV
    train_df.to_csv(configs['train_csv'], index=False)
    val_df.to_csv(configs['val_csv'], index=False)
    test_df.to_csv(configs['test_csv'], index=False)

def split_data_mimic_cxr(frac=0.05):
    """
    Split data into train, val, test, from mimic-cxr dataset
    """

    df = pd.read_csv(configs['mimic_csv_file_path'])
    df = df.sample(frac=frac, random_state=SEED)
    df.drop(columns='Unnamed: 0')
    print(f"Total rows: {df.shape[0]}")
    print(f"No Finding ratio: {df['No Finding'].mean()}")
    print(df.head())

    Y = df['report'].values

    train_df, val_df, test_df, _, _, _ = train_val_test_split(
        df, Y, configs['train_ratio'], configs['val_ratio'], configs['test_ratio'])

    # Write CSV
    train_df.to_csv(configs['train_csv'], index=False)
    val_df.to_csv(configs['val_csv'], index=False)
    test_df.to_csv(configs['test_csv'], index=False)

def main():
    dataset = configs['dataset']
    if dataset == 'iu-xray':
        split_data_iu_xray()
    elif dataset == 'mimic-cxr':
        split_data_mimic_cxr()

if __name__ == '__main__':
    main()