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

def split_data_mimic_cxr_old(frac=0.05):
    """
    [OBSOLETE]
    Split data into train, val, test, from mimic-cxr dataset
    """

    df = pd.read_csv(configs['mimic_csv_file_path'])
    if frac < 1.0:
        print(f"Sample ratio: {frac}")
        df = df.sample(frac=frac, random_state=SEED)
    if 'Unnamed: 0' in df.columns:
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

def split_data_mimic_cxr():
    """
    Split data into train, val, test according to 'mimic-cxr-2.0.0-split.csv'
    """

    df = pd.read_csv(configs['mimic_csv_file_path'])
    split_df = pd.read_csv(configs['mimic_dir'] + 'mimic-cxr-2.0.0-split.csv')
    print(f"Total rows: {df.shape[0]}")
    print(f"No Finding ratio: {df['No Finding'].mean()}")
    print(df.head())
    split = split_df['split']
    train_dicoms = split_df[split == 'train']['dicom_id']
    val_dicoms = split_df[split == 'validate']['dicom_id']
    test_dicoms = split_df[split == 'test']['dicom_id']
    print(f"Original split size: ({train_dicoms.shape[0]}, {val_dicoms.shape[0]}, {test_dicoms.shape[0]})")

    train_df = pd.merge(df, train_dicoms, how='inner', on=['dicom_id'])
    val_df = pd.merge(df, val_dicoms, how='inner', on=['dicom_id'])
    test_df = pd.merge(df, test_dicoms, how='inner', on=['dicom_id'])
    print(f"Post preprocess split size: ({train_df.shape[0]}, {val_df.shape[0]}, {test_df.shape[0]})")

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