import pickle
import os
import time
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.transform import resize
from tensorflow.keras.applications.densenet import preprocess_input
from tqdm import tqdm

from chexnet import ChexNet
from preprocess_text import remove_empty, text_preprocessing

from configs import configs

def add_start_stop(text) -> str:
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

    if os.path.exists(configs['images_npy_file_path']):
        print(f"Using cached images")
        images = np.load(configs['images_npy_file_path'])

    else:
        print(f"Loading {len(img_paths)} new images...")
        images = load_images(img_paths)
        
        print("Saving images...")
        np.save(configs['images_npy_file_path'], images)

    print(f"{images.shape=}")

    print(f"Generating image features..")
    image_features = generate_image_features(images)
    print(f"{image_features.shape=}")

    image_mappings = create_image_mappings(img_paths, image_features)
    return image_mappings


def save_csv(df) -> None:
    df.to_csv(configs['csv_file_path'], index=False)


def load_csv() -> pd.DataFrame:
    return pd.read_csv(configs['csv_file_path'])


def save_image_mappings(image_mappings) -> None:
    with open(configs['pickle_file_path'], 'wb') as f:
        print(f"Saving image mappings... ({len(image_mappings)} total images)")
        pickle.dump(image_mappings, f)


def load_image_mappings() -> dict:
    """
    Function for loading image mappings, for other python file to call
    """

    with open(configs['pickle_file_path'], 'rb') as f:
        image_mappings = pickle.load(f)
        return image_mappings

def read_report(path):
    """
    For mimic-cxr dataset
    """

    with open(path, 'r') as f:
        report = f.read()
        start = report.find('FINDINGS:')
        end = report.find('IMPRESSION:')
        report = report[start+9:end]
        report = report.replace('\n', ' ')
        report = ' '.join(report.split())

    return report

def load_dataframe(dataset) -> pd.DataFrame:
    """
    if dataset == 'iu-xray':
        Returns a joined dataframe consists of uid, filename, projection, and report.
        Using indiana_projections.csv and indiana_reports.csv
    elif dataset == 'mimic-cxr':

    """

    if dataset == 'iu-xray':

        COLUMNS_TO_RETURN = ['uid', 'img_path', 'Problems', 'projection', 'report']

        # Read the csv files
        report_df = pd.read_csv(configs['report_csv'])
        proj_df = pd.read_csv(configs['projection_csv'])

        df = pd.merge(proj_df, report_df, left_on='uid', right_on='uid')
        df['report'] = df['findings'].fillna('') + df['impression'].fillna('')

        # Full path
        df['img_path'] = configs['image_path'] + df['filename']
        df = df[COLUMNS_TO_RETURN]
        return df

    elif dataset == 'mimic-cxr':
        MIMIC_DIR = configs['mimic_dir']
        COLUMNS_TO_RETURN = ['dicom_id', 'subject_id', 'study_id', 
                        'ViewPosition', 'img_path', 'report_path', 'No Finding']

        record_df = pd.read_csv(MIMIC_DIR + 'cxr-record-list.csv')
        study_df = pd.read_csv(MIMIC_DIR + 'cxr-study-list.csv')

        merged = pd.merge(record_df, study_df, 
                            left_on=['subject_id', 'study_id'], 
                            right_on=['subject_id', 'study_id'])
        
        chexpert_df = pd.read_csv(MIMIC_DIR + 'mimic-cxr-2.0.0-chexpert.csv')[['subject_id', 'study_id', 'No Finding']]
        merged = pd.merge(merged, chexpert_df, 
                            left_on=['subject_id', 'study_id'], 
                            right_on=['subject_id', 'study_id'])
        
        merged['No Finding'] = merged['No Finding'].fillna(0.0)

        meta_df = pd.read_csv(MIMIC_DIR + 'mimic-cxr-2.0.0-metadata.csv')
        merged = pd.merge(meta_df, merged, 
                            left_on=['subject_id', 'study_id', 'dicom_id'], 
                            right_on=['subject_id', 'study_id', 'dicom_id'])

        # Rename columns
        merged = merged.rename(columns={'path_x': 'img_path', 'path_y': 'report_path'})

        # Filter columns
        merged = merged[COLUMNS_TO_RETURN]

        # Make full path
        merged['img_path'] = merged['img_path'].str.replace('files/', MIMIC_DIR + 'images512/')
        merged['img_path'] = merged['img_path'].str.replace('.dcm', '.png')
        merged['report_path'] = merged['report_path'].str.replace('files/', MIMIC_DIR + 'reports/files/')

        # Convert to report
        start_time = time.time()
        print(f"Total rows: {merged.shape[0]}")
        print("Converting report_path to report...")
        tqdm.pandas()
        merged['report'] = merged['report_path'].progress_apply(lambda x: read_report(x))
        print(f"Total time taken: {time.time() - start_time:.4f}s")

        # Drop missing reports
        merged = merged[merged['report'].str.len() > 0]
        merged = merged.reset_index(drop=True)

        # Drop rows start with FINAL
        merged = merged[~merged['report'].str.startswith('FINAL')]
        merged = merged.reset_index(drop=True)

        return merged

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

    print("Creating ChexNet model...")
    chexnet = ChexNet(input_shape=configs['input_shape'], pooling=None)
    image_features = []

    print("Creating image generator...")
    # batch
    # image_generator = tf.data.Dataset.from_tensor_slices(images) # Why explode here?
    # image_generator = image_generator.batch(batch_size)

    image_generator = []
    for i in range(0, images.shape[0], batch_size):
        start = i
        stop = min(images.shape[0], i + 32)
        image_generator.append(images[start:stop])

    print("Start generating features...")
    for batch in tqdm(image_generator):
        features_batch = chexnet(batch)
        image_features.extend(features_batch)

    return np.asarray(image_features)

def create_image_mappings(img_paths, image_features) -> dict:

    if len(img_paths) != image_features.shape[0]:
        raise Exception(f"{len(img_paths)} != {len(image_features.shape[0])}")

    return dict(zip(img_paths, image_features))

if __name__ == "__main__":
    # TODO: Add command line options to do either
    #       just image or report preprocessing
    dataset = configs['dataset']
    create_new = False
    
    if create_new:
        df = load_dataframe(dataset)

        # preprocess reports
        df = preprocess_text(df)
        print(f"Total report rows: {df.shape[0]}")

        print(df.head())
        print(df.columns)
        print(f"Total rows: {df.shape[0]}")
        print(f"No finding ratio: {df['No Finding'].mean()}")
        df.to_csv(configs['mimic_csv_file_path'])

    else:
        df = pd.read_csv(configs['mimic_csv_file_path'])
