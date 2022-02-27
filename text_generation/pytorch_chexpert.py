# import time
import warnings
from collections import OrderedDict
from pytorch_test import (TEMP_CSV_FILENAME, apply_logreg_mapping, load_unlabeled_data,
                  write_csv_for_chexbert)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy.lib.function_base import average
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import BertTokenizer
from utils import decode_sequences

import VisualCheXbert.visualchexbert.bert_tokenizer as bert_tokenizer
import VisualCheXbert.visualchexbert.utils as utils
from pytorch_tokenizer import create_tokenizer
from VisualCheXbert.visualchexbert.constants import *
from VisualCheXbert.visualchexbert.models.bert_labeler import bert_labeler
from configs import configs
import inspect

from dataset import ChestXRayDataset
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", category=UserWarning)

CHECKPOINT_FOLDER = 'checkpoint/'

model = bert_labeler()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
checkpoint_path = f"{CHECKPOINT_FOLDER}/visualCheXbert.pth"
bert_tok = BertTokenizer.from_pretrained('bert-base-uncased')

if torch.cuda.device_count() > 0:  # works even if only 1 GPU available
    frame = inspect.stack()[0]
    module = inspect.getmodule(frame[0])
    filename = module.__file__
    print(f"caller: {filename}")
    print("Creating Chexpert reward module...")
    print("Using", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)  # to utilize multiple GPU's
    model = model.to(device)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    checkpoint = torch.load(
        checkpoint_path, map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()
    for k, v in checkpoint['model_state_dict'].items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

def chexpert(sequences: np.array, tokenizer) -> pd.DataFrame:
    """
    Receives our model's sequence of int and tokenizer
    Returns precision, recall, or F1 of the labels

    Parameters:
        sequences (np.array): input array with sizes = (batch_size, max_len) where max_len is env's max_len
        tokenizer (tensorflow.keras.preprocessing.text.tokenizer): agent model's tokenizer

    Returns:
        pd.DataFrame: Dataframe containing 13 types of diagnosis for each column, where row numbers = batch_size
    """

    y_pred = [[] for _ in range(len(CONDITIONS))]

    # [[idx1, idx2, idx3, ...], [idx1, idx2, idx3, ...]] -> ['word1 word2 word3 ...', 'word1 word2 word3 ...']
    sentences = decode_sequences(tokenizer, sequences)

    # Tokenize with BertTokenizer
    imp = bert_tok.batch_encode_plus(sentences)['input_ids']

    # Limit length to 512 according to bert tokenizer
    imp = [torch.tensor(e) if len(e) <= 512 else torch.tensor(e[:511] + [bert_tok.sep_token_id]) for e in imp]
    imp = pad_sequence(imp, batch_first=True, padding_value=PAD_IDX)
    
    # Length for each sentence
    lengths = [len(e) for e in imp]
    # print(lengths)

    # Convert to torch tensor
    imp = torch.tensor(imp)
    imp = imp.to(device)

    # Generate attention masks
    attn_mask = utils.generate_attention_masks(imp, lengths, device)

    # Raw output
    out = model(imp, attn_mask)

    for j in range(len(out)):
        curr_y_pred = torch.sigmoid(out[j])  # shape is (batch_size)
        y_pred[j].append(curr_y_pred)

    for j in range(len(y_pred)):
        y_pred[j] = torch.cat(y_pred[j], dim=0)

    # Free GPU memory
    del imp
    torch.cuda.empty_cache()

    y_pred = [t.tolist() for t in y_pred]
    y_pred = np.array(y_pred)
    y_pred = y_pred.T
    df = pd.DataFrame(y_pred, columns=CONDITIONS)

    # Apply mapping from probs to image labels
    logreg_models_path = f"{CHECKPOINT_FOLDER}/logreg_models.pickle"
    df_visualchexbert = apply_logreg_mapping(df, logreg_models_path)

    # print(df_visualchexbert)
    return df_visualchexbert.drop(columns=['No Finding'])


def calculate_reward(ground_truth, prediction, tokenizer):
    df = chexpert(
        np.array([ground_truth, prediction]), tokenizer
    )

    t = df.iloc[0, :]
    p = df.iloc[1, :]

    t = t.values
    p = p.values

    # TODO: Possibly apply weight to each category if data is imbalance
    true_positive = np.logical_and(t, p)
    false_positive = np.logical_and(np.logical_not(t), p)
    false_negative = np.logical_and(t, np.logical_not(p))

    tp = np.count_nonzero(true_positive)
    fp = np.count_nonzero(false_positive)
    fn = np.count_nonzero(false_negative)

    if fp+tp == 0:
        precision = 0.0
    else:
        precision = tp/(fp+tp)
    
    if fn+tp == 0:
        recall = 0
    else:  
        recall = tp/(fn+tp)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1

def apply_labels_to_dataset(data_split, batch_size=12):
    tokenizer = create_tokenizer()
    data_loader = DataLoader(
        ChestXRayDataset(data_split),
        batch_size=batch_size,
        num_workers=1,
        shuffle=False,
        pin_memory=True,
    )

    labels = []
    for i, obj in enumerate(tqdm(data_loader)):
        caps = obj[1]
        label = chexpert(caps, tokenizer)
        labels.append(label)

    labels = pd.concat(labels).reset_index(drop=True)
    return labels

def apply_labels_to_csv():
    train_labels = apply_labels_to_dataset('train')
    train_labels.to_csv(configs['train_label_csv'], index=False)

    val_labels = apply_labels_to_dataset('val')
    val_labels.to_csv(configs['val_label_csv'], index=False)

    test_labels = apply_labels_to_dataset('test')
    test_labels.to_csv(configs['test_label_csv'], index=False)

def test_calculate_reward():
    import random
    # chexpert(np.array([7, 8, 9, 10, 4, 5, 0, 0]), tokenizer=cnn_rnn_tokenizer())
    tok = create_tokenizer()
    # sent1 = tok.texts_to_sequences(['bilateral patchy pulmonary opacities noted . interval improvement in left base consolidative opacity . pulmonary vascular congestion again noted . stable enlarged cardiomediastinal silhouette . stable left xxxx . no evidence of pneumothorax . no large pleural effusions . interval improvement in consolidative left base opacity . multifocal scattered bibasilar patchy and xxxx pulmonary opacities again noted most consistent with atelectasisinfiltrate . stable enlarged cardiomediastinal silhouette . stable pulmonary vascular congestion .'])
    # sent2 = tok.texts_to_sequences(
    #     ['lungs are overall hyperexpanded with flattening of the diaphragms . no focal consolidation . no pleural effusions or pneumothoraces . heart and mediastinum of normal size and contour . degenerative changes in the thoracic spine . hyperexpanded but clear lungs .'])
    # print(sent1)
    # print(sent2)
    sent1 = [[]]
    sent2 = [[]]
    max_len = 100
    vocab_size_fake = 500
    for i in range(max_len):
        id1 = random.randint(1, vocab_size_fake - 1)
        id2 = random.randint(1, vocab_size_fake - 1)
        sent1[0].append(id1)
        sent2[0].append(id2)
    
    print(sent1)
    print(sent2)

    for i in range(1):
        print(calculate_reward(
            np.array(sent1[0]),
            np.array(sent1[0]),
            tokenizer=tok
        ))

def test_chexpert():
    from dataset import ChestXRayDataset
    from torch.utils.data import DataLoader
    from pytorch_tokenizer import create_tokenizer

    tokenizer = create_tokenizer()
    data_loader = DataLoader(
        ChestXRayDataset('val'),
        batch_size=2,
        pin_memory=True,
    ) 

    _, captions, _, _ = next(iter(data_loader))

    # decoded_captions = tokenizer.decode(captions)
    df = chexpert(captions, tokenizer)
    print(df)

if __name__ == '__main__':
    apply_labels_to_csv()
