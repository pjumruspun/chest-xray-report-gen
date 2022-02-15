import pandas as pd
import numpy as np
from tqdm import tqdm
import time

from configs import configs
from preprocess_text import preprocess_text

def load_iu_xray_dataframe():
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
    df = df[COLUMNS_TO_RETURN]
    return df

def read_report_mimic_cxr(path):
    """
    Read and scan for FINDINGS and return text from FINDINGS until IMPRESSION
    """

    with open(path, 'r') as f:
        report = f.read()
        start = report.find('FINDINGS:')
        end = report.find('IMPRESSION:')
        report = report[start+9:end]
        report = report.replace('\n', ' ')
        report = ' '.join(report.split())

    return report

def load_mimic_cxr_dataframe():
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
    merged['report'] = merged['report_path'].progress_apply(lambda x: read_report_mimic_cxr(x))
    print(f"Total time taken: {time.time() - start_time:.4f}s")

    # Drop missing reports
    merged = merged[merged['report'].str.len() > 0]
    merged = merged.reset_index(drop=True)

    # Drop rows start with FINAL
    merged = merged[~merged['report'].str.startswith('FINAL')]
    merged = merged.reset_index(drop=True)

    return merged

def main():
    dataset = configs['dataset']

    if dataset == 'iu-xray':
        df = load_iu_xray_dataframe()

        # preprocess reports
        df = preprocess_text(df)
        print(f"Total report rows: {df.shape[0]}")

        # save to csv file
        df.to_csv(configs['iu_xray_csv_file_path'], index=False)

    elif dataset == 'mimic-cxr':
        df = load_mimic_cxr_dataframe(dataset)

        # preprocess reports
        df = preprocess_text(df)
        print(f"Total report rows: {df.shape[0]}")

        print(df.head())
        print(df.columns)
        print(f"Total rows: {df.shape[0]}")
        print(f"No finding ratio: {df['No Finding'].mean()}")
        df.to_csv(configs['mimic_csv_file_path'])

    else:
        raise ValueError("Wrong dataset received: {dataset}")

if __name__ == '__main__':
    main()