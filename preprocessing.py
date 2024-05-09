import re
import pandas as pd
import random
import logging

logging.basicConfig(level=logging.DEBUG)

def text_cleaning(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def text_preprocessing(lang, n=50000):
    # Read data
    path = f'./corpora/openSubtitles/raw_files/{lang}.txt'
    texts = open(path, mode='r', encoding='utf8').read().split('\n')
    # Remove punctuation and delete duplicate texts
    clean_texts = set([text_cleaning(text) for text in texts])
    clean_texts = sorted(list(clean_texts))
    print(lang, len(clean_texts))
    # select N random examples to build the datatset
    random.seed(42)
    clean_texts = random.sample(clean_texts, n)
    labels = [lang] * n
    return clean_texts, labels
    
def main():
    full_texts = []
    full_labels = []

    langs = ['en', 'es', 'fr', 'de', 'it', 'pt', 'tr', 'nl', 'sv']
    instances = [104153, 80540, 46445, 23743, 43661, 41187, 51388, 41221, 17662]

    for lang, ins in zip(langs, instances):
        logging.debug(f'Processing file: {lang}.txt')
        texts, labels = text_preprocessing(lang, ins)
        logging.debug(f'Instances: {len(texts)}')
        full_texts.extend(texts)
        full_labels.extend(labels)
    
    df = pd.DataFrame()
    df['text'] = full_texts
    df['lang'] = full_labels
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(f'./corpora/openSubtitles/openSubtitles.csv', encoding='utf8', index=False)

if __name__ == '__main__':
    main()