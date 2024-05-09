def create_char2idx(df):
    chars = []
    texts = df.text.values.tolist()
    for text in texts:
        chars.extend(list(text))
    chars = set(chars)
    chars = sorted(chars)
    char2idx = {char: idx for idx, char in enumerate(chars, 2)}
    char2idx['PAD'] = 0
    char2idx['UNK'] = 1
    return char2idx

def create_label2idx(df):
    labels = sorted(set(df.lang.values.tolist()))
    label2idx = {label : idx for idx, label in enumerate(labels,1)}
    return label2idx

def create_idx2label(df):
    labels = sorted(set(df.lang.values.tolist()))
    idx2label = {idx : label for idx, label in enumerate(labels,1)}
    return idx2label


