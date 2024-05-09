from torch.utils.data import Dataset
from dataset.item import Item
from dataset.sentence import Sentence

class LID_Dataset(Dataset):
    def __init__(self, df, char2idx, label2idx):
        super(LID_Dataset, self).__init__()
        self.data = df
        self.char2idx = char2idx
        self.label2idx = label2idx
    
    def __len__(self):
        return len(self.data)
    
    def text2charidx(self, text):
        chars = list(text)
        charidx = []
        for char in chars:
            if char in self.char2idx:
                charidx.append(self.char2idx[char])
            else:
                charidx.append(1)
        return charidx
    
    def labels2idx(self, label):
        labelidx = self.label2idx[label]
        one_hot = [0.0] * len(self.label2idx)
        one_hot[labelidx-1] = 1.0
        return one_hot
    
    def __getitem__(self, idx):
        text = self.data.text[idx]
        label = self.data.lang[idx]
        textidxs = self.text2charidx(text)
        labelidx = self.labels2idx(label)
        item = Item(Sentence(textidxs), labelidx)
        
        return item
