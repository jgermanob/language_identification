from sklearn.metrics import classification_report
import numpy as np
import torch
from utils.save_data import save_logits, save_report

def evaluate(model, test_dataloader, device, idx2label, args):
    model.eval()
    predictions = []
    gold_labels = []
    full_logits = []
    with torch.no_grad():
        for batch in test_dataloader:
            batch_x_char = batch['x_chars'].to(device)
            batch_x_char_len = batch['x_chars_len'].to(device)
            labels = batch['labels'].to(device)

            logits = model(batch_x_char, batch_x_char_len)
            full_logits.append(logits.cpu())

            prediction = np.argmax(logits.cpu(), axis=1).tolist()
            predictions.extend(prediction)
            gold_label = np.argmax(labels.cpu(), axis=1).tolist()
            gold_labels.extend(gold_label)
    
    predictions = [idx2label[pred+1] for pred in predictions]
    gold_labels = [idx2label[gold+1] for gold in gold_labels]

    report = classification_report(gold_labels, predictions, digits=4)

    report_dict = classification_report(gold_labels, predictions, digits=4, output_dict=True)

    save_report(report_dict, f'./reports/fold_{args.fold}/report_{args.fold}.json')
    save_logits(np.concatenate(full_logits, axis=0), f'./logits/fold_{args.fold}/logits_{args.fold}.npz')
    
    return report