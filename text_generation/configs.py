import os


def get_path(f, path):
    return os.path.join(os.path.dirname(f), path).replace('\\', '/')

dataset = 'iu-xray'

configs = {
    'dataset': dataset,
    'train_ratio': 0.75,
    'val_ratio': 0.10,
    'test_ratio': 0.15,
    'train_csv': os.path.join(os.path.dirname(__file__), 'data/train.csv') if dataset == 'iu-xray' else os.path.join(os.path.dirname(__file__), 'mimic_cxr/train.csv'),
    'val_csv': os.path.join(os.path.dirname(__file__), 'data/val.csv') if dataset == 'iu-xray' else os.path.join(os.path.dirname(__file__), 'mimic_cxr/val.csv'),
    'test_csv': os.path.join(os.path.dirname(__file__), 'data/test.csv') if dataset == 'iu-xray' else os.path.join(os.path.dirname(__file__), 'mimic_cxr/test.csv'),
    'image_path': get_path(__file__, 'data/images/images_normalized/'),
    'report_csv': get_path(__file__, 'data/indiana_reports.csv'),
    'projection_csv': get_path(__file__, 'data/indiana_projections.csv'),
    'images_npy_file_path': get_path(__file__, 'data/images.npy'),
    'csv_file_path': get_path(__file__, 'data/all.csv'),
    'iu_xray_csv_file_path': get_path(__file__, 'data/all.csv'),
    'mimic_csv_file_path': get_path(__file__, 'mimic_cxr_reports.csv'),
    'pickle_file_path': get_path(__file__, 'data/image_features.pickle'),
    'mimic_dir': 'mimic_cxr/',
    'input_shape': (256, 256, 3),
    'START_TOK': '<startseq>',
    'STOP_TOK': '<endseq>',
    'learning_rate': 1e-3,
    'encoder_lr': 5e-4,
    'decoder_lr': 5e-4,
    "embedding_dim": 200,
    "decoder_units": 80,
    'epochs': 25,
    'batch_size': 16,
    'test_batch_size': 32,
    'prediction_file_name': 'prediction.csv',
    'eval_matrix_file_name': 'eval_matrix.csv',
    'pretrained_emb_path': os.path.join(os.path.dirname(__file__), 'weights/pubmed2018_w2v_200D/pubmed2018_w2v_200D.bin'),
    'emb_matrix_path': os.path.join(os.path.dirname(__file__), 'weights/embedding_matrix.npy')
}
