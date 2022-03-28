# import pickle
# from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
# import torchvision.transforms as transforms
from sklearn.metrics import auc, f1_score, precision_score, recall_score
# from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
# from transformers import BertTokenizer

# import VisualCheXbert.visualchexbert.bert_tokenizer as bert_tokenizer
# import VisualCheXbert.visualchexbert.utils as utils
from dataset import MultiLabelDataset
from utils import evaluate_transform, CONDITIONS
# from VisualCheXbert.visualchexbert.constants import *
# from VisualCheXbert.visualchexbert.models.bert_labeler import bert_labeler


CHECKPOINT_FOLDER = 'checkpoint/'
TEMP_CSV_FILENAME = 'temp.csv'

def evaluation_matrix(true, pred):
    t = true.copy()
    p = pred.copy()
    if 'Report Impression' in t.columns:
        t = t.drop(['Report Impression'], axis=1)
    if 'Report Impression' in p.columns:
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

def predict_eval(encoder, data_loader, loss_fn):
    encoder.eval()
    losses = []
    results = []
    true = []
    
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(tqdm(data_loader)):
            imgs = imgs.cuda()
            labels = labels.cuda()

            _, probs = encoder(imgs)
            loss = loss_fn(probs, labels)
            losses.append(loss)
            true.append(labels)
            results.append(probs)

    val_loss = torch.stack(losses).mean()
    print(f"val_loss: {val_loss}")

    val_preds = torch.cat(results).reshape(-1, len(CONDITIONS))
    ground_truth = torch.cat(true).reshape(-1, len(CONDITIONS))
    return val_preds, ground_truth

def get_tpr_fpr(pred, true, threshold):
    p = pred.clone()
    t = true.clone()
    p = (p > threshold).int().cpu().numpy()
    t = t.int().cpu().numpy()

    tp = np.count_nonzero(np.logical_and(p == 1, t == 1), axis=0)
    tn = np.count_nonzero(np.logical_and(p == 0, t == 0), axis=0)
    fp = np.count_nonzero(np.logical_and(p == 1, t == 0), axis=0)
    fn = np.count_nonzero(np.logical_and(p == 0, t == 1), axis=0)
    
    tpr = np.zeros(len(CONDITIONS))
    fpr = np.zeros(len(CONDITIONS))

    for i in range(len(CONDITIONS)):
        if tp[i] + fn[i] == 0:
            tpr[i] = 0.0
        else:
            tpr[i] = tp[i] / (tp[i] + fn[i])
        
        if tn[i] + fp[i] == 0:
            fpr[i] = 0.0
        else:
            fpr[i] = fp[i] / (tn[i] + fp[i])

    return tpr, fpr

def roc_coords(pred, true):
    threshold_sweep = np.arange(0.000, 1.001, 0.001)
    tprs = []
    fprs = []
    
    # -1 as default value
    best_threshold = -np.ones(len(CONDITIONS))
    best_tpr = -np.ones(len(CONDITIONS))
    best_fpr = -np.ones(len(CONDITIONS))
    
    for t in threshold_sweep:
        tpr, fpr = get_tpr_fpr(pred, true, t)
        tprs.append(tpr)
        fprs.append(fpr)
        
        for i in range(len(CONDITIONS)):
            if 1-tpr[i] - fpr[i] > 0 and best_threshold[i] == -1: # -1 is default value
                best_tpr[i] = tpr[i]
                best_fpr[i] = fpr[i]
                best_threshold[i] = t
    
    tprs = np.vstack(tprs).T # (len(CONDITIONS), sample_size)
    fprs = np.vstack(fprs).T # (len(CONDITIONS), sample_size)

    return tprs, fprs, best_tpr, best_fpr, best_threshold

def plot_roc(val_preds, ground_truth):
    tprs, fprs, best_tpr, best_fpr, best_threshold = roc_coords(val_preds, ground_truth)
    aucs = np.zeros(len(CONDITIONS))
    plt.figure(figsize=(28, 8))
    
    for i in range(len(CONDITIONS)):
        x = np.arange(0, 1, 1/len(tprs[i]))
        y = np.arange(1, 0, -1/len(tprs[i]))
        area = auc(fprs[i], tprs[i])
        aucs[i] = area
        title = \
            f'{CONDITIONS[i]}\n' + \
            f'EER={best_tpr[i]:4f}\n' + \
            f'Best threshold = {best_threshold[i]:4f}\n' + \
            f'AUROC={area:.4f}'
        
        plt.subplot(2, 7, i+1)
        plt.title(title)
        plt.plot(x, y, 'r--')
        plt.plot(fprs[i], tprs[i])
        plt.plot(best_fpr[i], best_tpr[i], 'go')
    plt.tight_layout()
    plt.show()

    df = pd.DataFrame.from_dict({
        'Diagnosis': CONDITIONS,
        'EER': best_tpr,
        'Best Threshold': best_threshold,
        'AUROC': aucs,
    }).set_index('Diagnosis')

    return df

def plot_roc_multiple(val_preds, ground_truth, label):
    tprs, fprs, best_tpr, best_fpr, best_threshold = roc_coords(val_preds, ground_truth)
    aucs = np.zeros(len(CONDITIONS))
    plt.figure(figsize=(28, 8))
    
    for i in range(len(CONDITIONS)):
        x = np.arange(0, 1, 1/len(tprs[i]))
        y = np.arange(1, 0, -1/len(tprs[i]))
        area = auc(fprs[i], tprs[i])
        aucs[i] = area
        title = \
            f'{CONDITIONS[i]}\n' + \
            f'EER={best_tpr[i]:4f}\n' + \
            f'Best threshold = {best_threshold[i]:4f}\n' + \
            f'AUROC={area:.4f}'
        
        plt.subplot(2, 7, i+1)
        plt.title(title)
        plt.plot(x, y, 'r--')
        plt.plot(fprs[i], tprs[i], label=label)
        plt.plot(best_fpr[i], best_tpr[i], 'go')

    df = pd.DataFrame.from_dict({
        'Diagnosis': CONDITIONS,
        'EER': best_tpr,
        'Best Threshold': best_threshold,
        'AUROC': aucs,
    }).set_index('Diagnosis')

    return df

def evaluate_models(model_paths, val_loader, loss_fn):
    tprs_list = []
    fprs_list = []
    best_tpr_list = []
    best_fpr_list = []
    best_threshold_list = []
    aucs = -np.ones((len(model_paths), len(CONDITIONS)))
    for model_path in model_paths:
        checkpoint = torch.load(model_path)
        print(f"Evaluating {model_path} ... (epoch {checkpoint['epoch'] + 1})")
        encoder = checkpoint['encoder']
        val_preds, val_ground_truth = predict_eval(encoder, val_loader, loss_fn)
        
        tprs, fprs, best_tpr, best_fpr, best_threshold = roc_coords(val_preds, val_ground_truth)
        tprs_list.append(tprs)
        fprs_list.append(fprs)
        best_tpr_list.append(best_tpr)
        best_fpr_list.append(best_fpr)
        best_threshold_list.append(best_threshold)
    
    plt.figure(figsize=(42, 12))
    for i in range(len(CONDITIONS)):
        title = CONDITIONS[i]
        plt.subplot(2, 7, i+1)
        plt.title(title)
        x = np.arange(0, 1, 1/len(tprs_list[0][i]))
        y = np.arange(0, 1, 1/len(tprs_list[0][i]))
        y_rev = np.arange(1, 0, -1/len(tprs_list[0][i]))
        plt.plot(x, y_rev, 'r--')
        plt.plot(x, y, 'b--')
        for j, p in enumerate(model_paths):
            area = auc(fprs_list[j][i], tprs_list[j][i])
            aucs[j, i] = area
            label_name = p.split('/')[-1].split('.')[0]
            plt.plot(fprs_list[j][i], tprs_list[j][i], label=label_name)

    plt.legend()
    plt.tight_layout()
    plt.show()

    d = {
        'Diagnosis': CONDITIONS,
    }

    for j in range(len(model_paths)):
        d[f'model_{j+1}_AUROC'] = aucs[j]

    df = pd.DataFrame.from_dict(d).set_index('Diagnosis')
    df.loc['Mean'] = df.mean()
    return df

def evaluate_encoders(batch_size=16):
    model_paths = [
        # 'weights/pretrained_encoder/pretrained_enc_epoch_1_2022-03-01_21-25-43.558663.pth.tar',
        # 'weights/pretrained_encoder/pretrained_enc_epoch_2_2022-03-01_21-33-03.324598.pth.tar',
        # 'weights/pretrained_encoder/pretrained_enc_epoch_3_2022-03-01_21-40-23.618929.pth.tar',
        # 'weights/pretrained_encoder/pretrained_enc_epoch_4_2022-03-01_21-47-49.833802.pth.tar',
        # 'weights/pretrained_encoder/pretrained_enc_epoch_5_2022-03-01_21-55-18.022861.pth.tar',
        # 'weights/pretrained_encoder/pretrained_enc_epoch_10_2022-03-01_23-09-33.796416.pth.tar',
        # 'weights/pretrained_encoder/pretrained_enc_epoch_1_2022-03-07_15-05-00.074771.pth.tar',
        # 'weights/pretrained_encoder/pretrained_enc_epoch_5_2022-03-07_15-26-56.402211.pth.tar',
        # 'weights/pretrained_encoder/pretrained_enc_epoch_5_2022-03-07_16-26-30.081634.pth.tar',
        # 'weights/pretrained_encoder/pretrained_enc_epoch_10_2022-03-07_16-45-39.632080.pth.tar',
        # 'weights/pretrained_encoder/pretrained_enc_epoch_10_2022-03-07_17-32-49.918963.pth.tar', # Partial data 14
        # 'weights/pretrained_encoder/pretrained_enc_epoch_15_2022-03-07_17-51-34.444548.pth.tar' # Partial data 14
        # 'weights/pretrained_encoder/pretrained_enc_epoch_2_2022-03-08_11-10-45.680034.pth.tar', # Full data
        # 'weights/pretrained_encoder/pretrained_enc_epoch_5_2022-03-08_15-43-47.540586.pth.tar', # Full data current best
        # 'weights/pretrained_encoder/pretrained_enc_epoch_8_2022-03-09_02-35-44.194832.pth.tar',
        # 'weights/pretrained_encoder/pretrained_enc_epoch_10_2022-03-09_05-30-31.063956.pth.tar',
        'weights/pretrained_encoder/pretrained_enc_epoch_5_2022-03-17_02-59-19.803016.pth.tar', # Proper split
    ]

    val_loader = DataLoader(
        MultiLabelDataset('val', transform=evaluate_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    loss_fn = nn.BCELoss().cuda()

    result_df = evaluate_models(model_paths, val_loader, loss_fn)
    print(result_df)
    return result_df

def test_model(model_path, batch_size=16):
    test_loader = DataLoader(
        MultiLabelDataset('test', transform=evaluate_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    checkpoint = torch.load(model_path)
    encoder = checkpoint['encoder']
    loss_fn = nn.BCELoss().cuda()
    test_preds, test_ground_truth = predict_eval(encoder, test_loader, loss_fn)
    df = plot_roc(test_preds, test_ground_truth)
    df.loc['Mean'] = df.mean()
    return df

def main():
    model_to_test = 'weights/pretrained_encoder/pretrained_enc_epoch_5_2022-03-08_15-43-47.540586.pth.tar', # Proper split
    test_model(model_to_test)

if __name__ == '__main__':
    main()
