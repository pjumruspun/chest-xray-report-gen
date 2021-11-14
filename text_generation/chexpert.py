import time
import warnings
from collections import OrderedDict
from test import (TEMP_CSV_FILENAME, apply_logreg_mapping, load_unlabeled_data,
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

import VisualCheXbert.visualchexbert.bert_tokenizer as bert_tokenizer
import VisualCheXbert.visualchexbert.utils as utils
from tokenizer import cnn_rnn_tokenizer, decode_report
from VisualCheXbert.visualchexbert.constants import *
from VisualCheXbert.visualchexbert.models.bert_labeler import bert_labeler
from configs import configs

warnings.filterwarnings("ignore", category=UserWarning)

CHECKPOINT_FOLDER = 'checkpoint/'

model = bert_labeler()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
checkpoint_path = f"{CHECKPOINT_FOLDER}/visualCheXbert.pth"
bert_tok = BertTokenizer.from_pretrained('bert-base-uncased')

if torch.cuda.device_count() > 0:  # works even if only 1 GPU available
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

# model.eval() # What does this do?
sent_decode_time = []
bert_tok_time = []
tensor_convert_time = []
gen_mask_time = []
model_time = []
post_process_time = []
logreg_model_time = []


def chexpert(sequences, tokenizer):
    """
    Receives our model's sequence of int and tokenizer
    Returns precision, recall, or F1 of the labels
    """

    y_pred = [[] for _ in range(len(CONDITIONS))]

    start = time.time()
    # Decode sequence into string
    # sentences = decode_report(tokenizer, np.array(sequences))
    sentences = tokenizer.sequences_to_texts(np.array(sequences))
    new_sentences = []
    for sent in sentences:
        new_sentences.append(sent.replace('<unk>', '').replace(configs['START_TOK'], '').replace(configs['STOP_TOK'], '').strip())
    sentences = new_sentences
    print(sentences)

    end = time.time()
    sent_decode_time.append(end - start)
    start = time.time()

    # Tokenize with BertTokenizer

    imp = bert_tok.batch_encode_plus(sentences)['input_ids']
    imp = [torch.tensor(e) for e in imp]
    imp = pad_sequence(imp, batch_first=True, padding_value=PAD_IDX)
    # print(f"{imp}=")

    end = time.time()

    bert_tok_time.append(end - start)
    lengths = [len(e) for e in imp]

    # print(f"{lengths}")

    # Convert to torch tensor
    start = time.time()
    imp = torch.tensor(imp)
    imp = imp.to(device)
    end = time.time()
    tensor_convert_time.append(end - start)
    # print(f"{imp.shape=}")

    start = time.time()
    # Generate attention masks
    attn_mask = utils.generate_attention_masks(imp, lengths, device)
    end = time.time()
    gen_mask_time.append(end - start)

    # Raw output
    start = time.time()
    out = model(imp, attn_mask)
    end = time.time()
    model_time.append(end - start)
    # print(f"{out=}")

    start = time.time()
    for j in range(len(out)):
        curr_y_pred = torch.sigmoid(out[j])  # shape is (batch_size)
        y_pred[j].append(curr_y_pred)

    for j in range(len(y_pred)):
        y_pred[j] = torch.cat(y_pred[j], dim=0)

    y_pred = [t.tolist() for t in y_pred]
    y_pred = np.array(y_pred)
    y_pred = y_pred.T
    df = pd.DataFrame(y_pred, columns=CONDITIONS)
    # print(df)
    # print(y_pred)
    end = time.time()
    post_process_time.append(end - start)

    start = time.time()
    # Apply mapping from probs to image labels
    logreg_models_path = f"{CHECKPOINT_FOLDER}/logreg_models.pickle"
    df_visualchexbert = apply_logreg_mapping(df, logreg_models_path)
    end = time.time()
    logreg_model_time.append(end - start)
    # print(df_visualchexbert)
    return df_visualchexbert


def calculate_reward(ground_truth, prediction, tokenizer):
    df = chexpert(
        np.array([ground_truth, prediction]), tokenizer
    )

    t = df.iloc[0, :]
    p = df.iloc[1, :]

    t = t.values
    p = p.values

    # TODO: Possibly apply weight to each category if data is imbalance
    # print(t)
    # print(p)
    true_positive = np.logical_and(t, p)
    false_positive = np.logical_and(np.logical_not(t), p)
    false_negative = np.logical_and(t, np.logical_not(p))
    # print(true_positive)
    # print(false_positive)
    # print(false_negative)

    tp = np.count_nonzero(true_positive)
    fp = np.count_nonzero(false_positive)
    fn = np.count_nonzero(false_negative)

    # print(tp, fp, fn)

    precision = tp/(fp+tp)
    recall = tp/(fn+tp)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    # print(precision, recall, f1)

    return precision, recall, f1


if __name__ == '__main__':
    # chexpert(np.array([7, 8, 9, 10, 4, 5, 0, 0]), tokenizer=cnn_rnn_tokenizer())
    tok = cnn_rnn_tokenizer()
    sent1 = tok.texts_to_sequences(['bilateral patchy pulmonary opacities noted . interval improvement in left base consolidative opacity . pulmonary vascular congestion again noted . stable enlarged cardiomediastinal silhouette . stable left xxxx . no evidence of pneumothorax . no large pleural effusions . interval improvement in consolidative left base opacity . multifocal scattered bibasilar patchy and xxxx pulmonary opacities again noted most consistent with atelectasisinfiltrate . stable enlarged cardiomediastinal silhouette . stable pulmonary vascular congestion .'])
    sent2 = tok.texts_to_sequences(
        ['lungs are overall hyperexpanded with flattening of the diaphragms . no focal consolidation . no pleural effusions or pneumothoraces . heart and mediastinum of normal size and contour . degenerative changes in the thoracic spine . hyperexpanded but clear lungs .'])
    # print(sent1)
    # print(sent2)
    for i in range(1):
        print(calculate_reward(
            np.array(sent1[0]),
            np.array(sent1[0]),
            tokenizer=tok
        ))

    # print(f"{sent_decode_time=}")
    # print(f"{bert_tok_time=}")
    # print(f"{tensor_convert_time=}")
    # print(f"{gen_mask_time=}")
    # print(f"{model_time=}")
    # print(f"{post_process_time=}")
    # print(f"{logreg_model_time=}")

    # print(f"{average(sent_decode_time)=}")
    # print(f"{average(bert_tok_time)=}")
    # print(f"{average(tensor_convert_time)=}")
    # print(f"{average(gen_mask_time)=}")
    # print(f"{average(model_time)=}")
    # print(f"{average(post_process_time)=}")
    # print(f"{average(logreg_model_time)=}")
