from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from numpy.core.defchararray import decode
import torch
import torch.nn as nn
import pickle
import pandas as pd
import numpy as np

from tqdm import tqdm
from dataset import get_train_materials
from label import generate_sentence, prep_models, prettify

import VisualCheXbert.visualchexbert.utils as utils

import VisualCheXbert.visualchexbert.bert_tokenizer as bert_tokenizer
from VisualCheXbert.visualchexbert.models.bert_labeler import bert_labeler
from VisualCheXbert.visualchexbert.bert_tokenizer import tokenize
from transformers import BertTokenizer
from collections import OrderedDict
from VisualCheXbert.visualchexbert.constants import *

from torch.utils.data import Dataset
from configs import configs
from numba import cuda 

CHECKPOINT_FOLDER = 'checkpoint/'
TEMP_CSV_FILENAME = 'temp.csv'


def prettify(raw_text, delimiter=' '):
    res = []
    for sentence in raw_text.split('.'):
        if len(sentence.strip()) == 0:
            continue
        sentence = sentence.replace('<startseq>', '')
        sentence = sentence.replace('<endseq>', '')
        res.append(sentence.strip().capitalize() + '.')
    return delimiter.join(res)


def write_csv_for_chexbert(reports):
    """Write temporary .csv file for VisualCheXbert

    """
    f = open(TEMP_CSV_FILENAME, 'w')
    f.write("Report Impression\n")
    for r in reports:
        prettified = prettify(r)
        prettified = prettified.replace('"', '')
        prettified = prettified.replace(',', '')
        prettified = prettified.strip()
        f.write(prettified + '\n')
    f.close()


class UnlabeledDataset(Dataset):
    """The dataset to contain report impressions without any labels."""

    def __init__(self, csv_path):
        """ Initialize the dataset object
        @param csv_path (string): path to the csv file containing rhe reports. It
                                  should have a column named "Report Impression"
        """
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        impressions = bert_tokenizer.get_impressions_from_csv(csv_path)
        self.encoded_imp = bert_tokenizer.tokenize(impressions, tokenizer)

    def __len__(self):
        """Compute the length of the dataset

        @return (int): size of the dataframe
        """
        return len(self.encoded_imp)

    def __getitem__(self, idx):
        """ Functionality to index into the dataset
        @param idx (int): Integer index into the dataset

        @return (dictionary): Has keys 'imp', 'label' and 'len'. The value of 'imp' is
                              a LongTensor of an encoded impression. The value of 'label'
                              is a LongTensor containing the labels and 'the value of
                              'len' is an integer representing the length of imp's value
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        imp = self.encoded_imp[idx]
        imp = torch.LongTensor(imp)
        return {"imp": imp, "len": imp.shape[0]}


def collate_fn_no_labels(sample_list):
    """Custom collate function to pad reports in each batch to the max len,
       where the reports have no associated labels
    @param sample_list (List): A list of samples. Each sample is a dictionary with
                               keys 'imp', 'len' as returned by the __getitem__
                               function of ImpressionsDataset
    @returns batch (dictionary): A dictionary with keys 'imp' and 'len' but now
                                 'imp' is a tensor with padding and batch size as the
                                 first dimension. 'len' is a list of the length of 
                                 each sequence in batch
    """
    tensor_list = [s['imp'] for s in sample_list]
    batched_imp = torch.nn.utils.rnn.pad_sequence(tensor_list,
                                                  batch_first=True,
                                                  padding_value=PAD_IDX)
    len_list = [s['len'] for s in sample_list]
    batch = {'imp': batched_imp, 'len': len_list}
    return batch


def load_unlabeled_data(csv_path, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                        shuffle=False):
    """ Create UnlabeledDataset object for the input reports
    @param csv_path (string): path to csv file containing reports
    @param batch_size (int): the batch size. As per the BERT repository, the max batch size
                             that can fit on a TITAN XP is 6 if the max sequence length
                             is 512, which is our case. We have 3 TITAN XP's
    @param num_workers (int): how many worker processes to use to load data
    @param shuffle (bool): whether to shuffle the data or not  

    @returns loader (dataloader): dataloader object for the reports
    """
    collate_fn = collate_fn_no_labels
    dset = UnlabeledDataset(csv_path)
    loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle,
                                         num_workers=num_workers, collate_fn=collate_fn)
    return loader


def apply_logreg_mapping(df_probs, logreg_models_path):
    logreg_models = {}
    visualchexbert_dict = {}
    try:
        with open(logreg_models_path, "rb") as handle:
            logreg_models = pickle.load(handle)
    except Exception as e:
        print("Error loading path to logistic regression models. Please ensure that the pickle file is in the checkpoint folder.")
        print(f"Exception: {e}")
    for condition in CONDITIONS:
        clf = logreg_models[condition]
        y_pred = clf.predict(df_probs)
        visualchexbert_dict[condition] = y_pred
    df_visualchexbert = pd.DataFrame.from_dict(visualchexbert_dict)
    return df_visualchexbert


def label(reports, out_path=None):
    """Labels a dataset of reports
    @param checkpoint_path (string): location of saved model checkpoint 
    @param csv_path (string): location of csv with reports
    @param out_path (string): path to output directory
    @returns y_pred (List[List[int]]): Labels for each of the 14 conditions, per report  
    """

    checkpoint_folder = CHECKPOINT_FOLDER
    write_csv_for_chexbert(reports)
    ld = load_unlabeled_data(TEMP_CSV_FILENAME)

    model = bert_labeler()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint_path = f"{checkpoint_folder}/visualCheXbert.pth"
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

    was_training = model.training
    model.eval()
    y_pred = [[] for _ in range(len(CONDITIONS))]

    print("\nBegin report impression labeling. The progress bar counts the # of batches completed:")
    print("The batch size is %d" % BATCH_SIZE)
    with torch.no_grad():
        for i, data in enumerate(tqdm(ld)):
            batch = data['imp']  # (batch_size, max_len)
            batch = batch.to(device)
            src_len = data['len']
            batch_size = batch.shape[0]
            attn_mask = utils.generate_attention_masks(batch, src_len, device)

            out = model(batch, attn_mask)

            for j in range(len(out)):
                curr_y_pred = torch.sigmoid(out[j])  # shape is (batch_size)
                y_pred[j].append(curr_y_pred)

        for j in range(len(y_pred)):
            y_pred[j] = torch.cat(y_pred[j], dim=0)

        print(f"{len(y_pred)=}")
        print(f"{y_pred[0]=}")
        print(f"{y_pred[1]=}")

    if was_training:
        model.train()

    y_pred = [t.tolist() for t in y_pred]
    y_pred = np.array(y_pred)
    y_pred = y_pred.T

    df = pd.DataFrame(y_pred, columns=CONDITIONS)

    # Apply mapping from probs to image labels
    logreg_models_path = f"{checkpoint_folder}/logreg_models.pickle"
    df_visualchexbert = apply_logreg_mapping(df, logreg_models_path)

    reports = pd.read_csv(TEMP_CSV_FILENAME)['Report Impression'].tolist()

    df_visualchexbert.insert(loc=0, column='Report Impression', value=reports)

    # Save to CSV
    # df_visualchexbert.to_csv(os.path.join(out_path, 'labeled_reports.csv'), index=False)

    return df_visualchexbert


def decode_report(tokenizer, arr):
    arr = np.asarray(arr)
    res = [[] for _ in range(arr.shape[0])]

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i][j] != 0:
                word = tokenizer.index_word[arr[i][j]]
                if word != configs['START_TOK'] and word != configs['STOP_TOK']:
                    res[i].append(word)
        res[i] = ' '.join(res[i])

    return res


def evaluation_matrix(true, pred):
    t = true.copy()
    p = pred.copy()
    t = t.drop(['Report Impression'], axis=1)
    p = p.drop(['Report Impression'], axis=1)
    # accuracy = ['Accuracy']
    # for col in t.columns:
    #   accuracy.append(accuracy_score(t[col], p[col]))

    recall = ['Recall']
    for col in t.columns:
        recall.append(recall_score(t[col], p[col]))

    recall.append(recall_score(t, p, average='macro'))
    recall.append(recall_score(t, p, average='micro'))

    precision = ['Precision']
    for col in t.columns:
        precision.append(precision_score(t[col], p[col]))

    precision.append(precision_score(t, p, average='macro'))
    precision.append(precision_score(t, p, average='micro'))

    f1 = ['F1']
    for col in t.columns:
        f1.append(f1_score(t[col], p[col]))

    f1.append(f1_score(t, p, average='macro'))
    f1.append(f1_score(t, p, average='micro'))

    res = pd.DataFrame([
        # accuracy,
        recall,
        precision,
        f1
    ], columns=['Metrics'] + list(t.columns) + ['Macro', 'Micro'])
    res = res.set_index(['Metrics'])

    return res.T


def main():
    print("Preparing test generator...")
    generators, _, tokenizer, embedding_matrix, _ = get_train_materials()
    _, _, test_generator = generators

    print("Preparing models...")
    encoder, decoder = prep_models(embedding_matrix)

    results = {'ground_truth': [], 'prediction': []}

    # predict
    print(f"Generating results with batch_size = {configs['test_batch_size']}")
    for img_tensor, ground_truth in tqdm(test_generator):
        batch_size = img_tensor.shape[0]
        pred_report = generate_sentence(
            encoder, decoder, tokenizer, img_tensor, batch_size)
        results['prediction'].extend(pred_report)

        # decode ground truth
        gt = decode_report(tokenizer, ground_truth)
        results['ground_truth'].extend(gt)

    df = pd.DataFrame(results)
    df.to_csv(configs['prediction_file_name'])

    device = cuda.get_current_device()
    device.reset()

    # label with visualchexbert
    ground_truth_labeled = label(
        df['ground_truth'].apply(lambda x: prettify(x)).values)
    prediction_labeled = label(
        df['prediction'].apply(lambda x: prettify(x)).values)
    
    # shows results in evaluation matrix
    eval_matrix = evaluation_matrix(ground_truth_labeled, prediction_labeled)
    print(eval_matrix)
    eval_matrix.to_csv(configs['eval_matrix_file_name'])


if __name__ == '__main__':
    main()
