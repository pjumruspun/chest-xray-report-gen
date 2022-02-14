from skimage import transform
from sklearn.metrics import recall_score, precision_score, f1_score
import torch
import torch.nn as nn
import pickle
import pandas as pd
import numpy as np
from torch.utils.data.dataloader import DataLoader
from utils import decode_sequences

from tqdm import tqdm
from label import prettify
from pytorch_label import temperature_sampling

import VisualCheXbert.visualchexbert.utils as utils

import VisualCheXbert.visualchexbert.bert_tokenizer as bert_tokenizer
from VisualCheXbert.visualchexbert.models.bert_labeler import bert_labeler
from transformers import BertTokenizer
from collections import OrderedDict
from VisualCheXbert.visualchexbert.constants import *

from torch.utils.data import Dataset
from configs import configs
from numba import cuda 

from tokenizer import decode_report

from pytorch_dataset import ChestXRayDataset
from pytorch_tokenizer import create_tokenizer
import torchvision.transforms as transforms

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

        # print(f"{len(y_pred)=}")
        # print(f"{y_pred[0]=}")
        # print(f"{y_pred[1]=}")

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

    return df_visualchexbert

def evaluation_matrix(true, pred):
    t = true.copy()
    p = pred.copy()
    t = t.drop(['Report Impression'], axis=1)
    p = p.drop(['Report Impression'], axis=1)

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
        recall,
        precision,
        f1
    ], columns=['Metrics'] + list(t.columns) + ['Macro', 'Micro'])
    res = res.set_index(['Metrics'])

    return res.T


def main():
    checkpoint_paths = [
        # "weights\pytorch_attention\checkpoint_2021-12-13_21-04-41.632539.pth.tar",
        # "weights\pytorch_attention\checkpoint_2021-12-13_21-29-00.708475.pth.tar",
        # "weights\pytorch_attention\checkpoint_2021-12-13_21-52-35.726799.pth.tar",
        # "weights\pytorch_attention\checkpoint_2021-12-13_22-23-05.938944.pth.tar",
        # "weights\pytorch_attention\checkpoint_2021-12-13_22-47-18.564559.pth.tar",
        # "weights\pytorch_attention\checkpoint_2021-12-13_23-11-15.179474.pth.tar",
        # "weights\pytorch_attention\checkpoint_2021-12-13_23-34-31.115275.pth.tar",
        # "weights\pytorch_attention\checkpoint_2021-12-13_23-57-37.200092.pth.tar",
        # "weights\pytorch_attention\checkpoint_2021-12-14_00-20-49.084023.pth.tar",
        # "weights\pytorch_attention\checkpoint_2021-12-14_00-43-58.777496.pth.tar",
        # "weights\pytorch_attention\checkpoint_2021-12-14_01-07-09.016551.pth.tar", # <- 0.22 micro f1
        # "weights\pytorch_attention\checkpoint_2021-12-14_01-30-12.054247.pth.tar",
        # "weights\pytorch_attention\checkpoint_2021-12-14_01-53-22.372573.pth.tar",
        # "weights\pytorch_attention\checkpoint_2021-12-14_02-16-39.428884.pth.tar",
        # "weights\pytorch_attention\checkpoint_2021-12-14_02-39-45.241841.pth.tar",
        # "weights\pytorch_attention\checkpoint_2021-12-14_12-05-27.843526.pth.tar",
        # "weights\pytorch_attention\checkpoint_2021-12-14_12-28-11.592583.pth.tar",
        # "weights\pytorch_attention\checkpoint_2021-12-14_12-50-48.187843.pth.tar",
        # "weights\pytorch_attention\checkpoint_2021-12-14_14-23-42.430392.pth.tar",
        # 'weights/pytorch_attention/checkpoint_2022-01-07_17-44-48.868114.pth.tar', # <- RL
        # 'weights/pytorch_attention/checkpoint_2022-01-18_19-05-33.048100.pth.tar', # <- working RL
        'weights\pytorch_attention\checkpoint_2022-02-08_01-29-52.625176.pth.tar', # <- cleaned epoch 20 0.2957 macro f1
        'weights\pytorch_attention\checkpoint_2022-02-08_10-37-10.316094.pth.tar', # <- cleaned RL epoch 3
    ]

    temperatures = [1.0]
    batch_size = 32
    print("Preparing test generator...")

    tokenizer = create_tokenizer()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_loader = DataLoader(
        ChestXRayDataset('test', transform=transforms.Compose([normalize])),
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    first_time = True

    for checkpoint_path in checkpoint_paths:
        for temperature in temperatures:
            for attempt in range(5):
                print("Preparing models...")
                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                checkpoint = torch.load(checkpoint_path)
                encoder = checkpoint['encoder']
                decoder = checkpoint['decoder']
                epoch = checkpoint['epoch'] + 1
                print(f"Using epoch {epoch} model...")
                encoder.to(device)
                decoder.to(device)

                if first_time:
                    results = {'ground_truth': [], 'prediction': []}
                else:
                    results = {'ground_truth': results['ground_truth'], 'prediction': []}

                # predict
                print(f"Generating results with batch_size = {batch_size}")
                for i, (imgs, caps, caplens, _) in enumerate(tqdm(test_loader)):
                    seqs, _ = temperature_sampling(encoder, decoder, tokenizer, images=imgs, temperature=temperature, max_len=100)
                    sentences = decode_sequences(tokenizer, seqs)
                    results['prediction'].extend(sentences)

                    # decode ground truth
                    if first_time:
                        gts = decode_sequences(tokenizer, caps)
                        results['ground_truth'].extend(gts)

                unique_name = f"{checkpoint_path}_{attempt+1}"

                df = pd.DataFrame(results)
                df.to_csv(unique_name + '_' + configs['prediction_file_name'])

                # label with visualchexbert
                prediction_labeled = label(df['prediction'].apply(lambda x: prettify(x)).values)
                if first_time:
                    ground_truth_labeled = label(df['ground_truth'].apply(lambda x: prettify(x)).values)

                # mean of each disease
                mean_df = pd.DataFrame({
                    'prediction': prediction_labeled.mean(),
                    'ground_truth': ground_truth_labeled.mean(),
                })
                print(mean_df)
                mean_df.to_csv(unique_name + '_' + 'mean.csv')
                
                # shows results in evaluation matrix
                eval_matrix = evaluation_matrix(ground_truth_labeled, prediction_labeled)
                print(eval_matrix)
                eval_matrix.to_csv(unique_name + '_' + configs['eval_matrix_file_name'])

                first_time = False

if __name__ == '__main__':
    main()
