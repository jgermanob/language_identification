import numpy as np
import pandas as pd
from utils.create_dict import create_idx2label, create_char2idx, create_label2idx
from sklearn.metrics import classification_report
from dataset.dataset import LID_Dataset
from torch.utils.data import DataLoader
from dataset.collate import collate_fn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def get_ensemble_predictions(k, idx2label):
    full_logits = [np.load(f'./logits/fold_{i+1}/logits_{i+1}.npz')['arr_0'] for i in range(k)]
    full_logits = sum(full_logits) / k
    ensemble_predictions = np.argmax(full_logits, axis=1)
    predictions = [idx2label[pred+1] for pred in ensemble_predictions]
    
    return predictions

def get_gold_labels(df, char2idx, label2idx, idx2label):
    test_dataset = LID_Dataset(df, char2idx, label2idx)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    predictions = []
    for batch in test_dataloader:
        labels = batch['labels']
        prediction = np.argmax(labels, axis=1).tolist()
        predictions.extend(prediction)
    
    predictions = [idx2label[idx+1] for idx in predictions]

    return predictions


def main():
    train_df = pd.read_csv(f'./corpora/openSubtitles/train.csv', encoding='utf8')
    test_df = pd.read_csv(f'./corpora/openSubtitles/test.csv', encoding='utf8')
    
    char2idx = create_char2idx(train_df)
    label2idx = create_label2idx(train_df)
    idx2label = create_idx2label(train_df)

    predictions = get_ensemble_predictions(5, idx2label)
    gold_labels = get_gold_labels(test_df, char2idx, label2idx, idx2label)

    report = classification_report(gold_labels, predictions, digits=4)
    cm = confusion_matrix(gold_labels, predictions)
    disp = ConfusionMatrixDisplay(cm, display_labels=label2idx.keys())
    disp.plot()
    plt.savefig('confusion_matrix.png')

    #print(report)

if __name__=='__main__':
    main()