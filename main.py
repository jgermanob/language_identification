import pandas as pd
from dataset.dataset import LID_Dataset
from dataset.collate import collate_fn
from torch.utils.data import DataLoader
import argparse
from utils.create_dict import create_char2idx, create_label2idx, create_idx2label
from model.lstm import LID_LSTM
import torch
from trainer.trainer import LID_LSTM_Trainer
from torch import cuda
import logging
from evaluation.test import evaluate

logging.basicConfig(level=logging.DEBUG)

device = 'cuda' if cuda.is_available() else 'cpu'

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=1)
    # Model arguments
    parser.add_argument('--embedding_dim', type=int, default=150)
    parser.add_argument('--hidden_dim', type=int, default=150)
    parser.add_argument('--layers', type=int, default=2)
    #Training arguments
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=5)
    args = parser.parse_args()
    return args

def load_data(args):
    train_path = f'./corpora/openSubtitles/folds/fold_{args.fold}/train.csv'
    val_path = f'./corpora/openSubtitles/folds/fold_{args.fold}/val.csv'
    full_train_path = './corpora/openSubtitles/train.csv'
    test_path = './corpora/openSubtitles/test.csv'

    df_train = pd.read_csv(train_path, encoding='utf8')
    df_val = pd.read_csv(val_path, encoding='utf8')
    df_test = pd.read_csv(test_path, encoding='utf8')

    df_full_train = pd.read_csv(full_train_path, encoding='utf8')
    char2idx = create_char2idx(df_full_train)
    label2idx = create_label2idx(df_full_train)
    idx2label = create_idx2label(df_full_train)

    train_dataset = LID_Dataset(df_train, char2idx, label2idx)
    val_dataset = LID_Dataset(df_val, char2idx, label2idx)
    test_dataset = LID_Dataset(df_test, char2idx, label2idx)

    vocab_size = len(char2idx)
    num_labels = len(label2idx)
    
    return train_dataset, val_dataset, test_dataset, vocab_size, num_labels, idx2label

def print_args(args):
    logging.debug(f'Fold = {args.fold}')
    logging.debug(f'Model parameters:\n\tEmbedding dim = {args.embedding_dim}\n\tLSTM hidden dim = {args.hidden_dim}\n\tLSTM layers = {args.layers}')
    logging.debug(f'Training parameters:\n\tEpochs = {args.epochs}\n\tBatch size = {args.batch_size}\n\tLearning rate = {args.learning_rate}\n\tPatience = {args.patience}')

def main():
    args = get_arguments()
    train_dataset, val_dataset, test_dataset, vocab_size, num_labels, idx2label = load_data(args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    print_args(args)
    model = LID_LSTM(vocab_size= vocab_size, embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim, layers=args.layers, num_classes=num_labels)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_function = torch.nn.CrossEntropyLoss()
    trainer = LID_LSTM_Trainer(args, optimizer, loss_function)
    
    model, best_model, history = trainer.train(model=model,
                                               train_dataloader=train_dataloader,
                                               val_dataloader=val_dataloader,
                                               test_dataloader=test_dataloader,
                                               device=device)
    
    report = evaluate(best_model, test_dataloader, device, idx2label, args)
    
    logging.debug(f'\n{report}')

if __name__ == '__main__':
    main()
