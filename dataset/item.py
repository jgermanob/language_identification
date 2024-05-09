from dataset.sentence import Sentence

class Item:
    def __init__(self, input: Sentence, output):
        self.input = input
        self.output = output
    
    def __len__(self):
        return len(self.input)