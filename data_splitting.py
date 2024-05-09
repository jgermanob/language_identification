import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import os

def create_train_test_split(df):
    texts = df.text.values.tolist()
    labels = df.lang.values.tolist()
    x_train, x_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)
    
    train_df = pd.DataFrame()
    train_df['text'] = x_train
    train_df['lang'] = y_train

    test_df = pd.DataFrame()
    test_df['text'] = x_test
    test_df['lang'] = y_test

    train_df.to_csv('./corpora/openSubtitles/train.csv', encoding='utf8', index=False)
    test_df.to_csv('./corpora/openSubtitles/test.csv', encoding='utf8', index=False)

def create_k_folds(df, k=5):
    texts = df.text
    labels = df.lang
    k_folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    for i, (train_index, val_index) in enumerate(k_folds.split(texts, labels)):
        print(f'Fold: {i}')
        x_train = texts[train_index]
        y_train = labels[train_index]
        x_val = texts[val_index]
        y_val = labels[val_index]
        #Creating folder to save the fold split
        folder_path = f'./corpora/openSubtitles/folds/fold_{i+1}'
        os.system(f'mkdir {folder_path}')
        
        #Saving train fold fata
        df_train = pd.DataFrame()
        df_train['text'] = x_train
        df_train['lang'] = y_train
        df_train.to_csv(f'{folder_path}/train.csv', encoding='utf8', index=False)
        
        #Saving validation fold data
        df_val = pd.DataFrame()
        df_val['text'] = x_val
        df_val['lang'] = y_val
        df_val.to_csv(f'{folder_path}/val.csv', encoding='utf8', index=False)

def main():
    df_path = './corpora/openSubtitles/openSubtitles.csv' 
    df = pd.read_csv(df_path, encoding='utf8')
    create_train_test_split(df)
    df_train_path = './corpora/openSubtitles/train.csv'
    df_train = pd.read_csv(df_train_path, encoding='utf8')
    create_k_folds(df_train)

if __name__ == '__main__':
    main()




        


