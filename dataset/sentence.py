from typing import List

class Sentence:
    def __init__(self, chars: List[int]):
        self.chars = chars
    
    def __len__(self):
        return len(self.chars)