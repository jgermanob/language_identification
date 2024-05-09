import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LID_LSTM(torch.nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, layers, num_classes):
        super(LID_LSTM, self).__init__()
        self.char_embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.lstm = torch.nn.LSTM(input_size=embedding_dim, hidden_size=int(hidden_dim/2), num_layers=layers, bidirectional=True, batch_first=True)
        self.fc = torch.nn.Linear(in_features=hidden_dim, out_features=num_classes)
        self.dropout = torch.nn.Dropout(p=0.5)
    
    def forward(self, input, input_len):
        x = self.char_embedding(input)
        x = self.dropout(x)
        
        sorted_x_len, perm_idx = input_len.sort(0, descending=True)
        _, recover_idx = perm_idx.sort(0, descending=False)
        sorted_x = x[perm_idx]
        
        pack_input = pack_padded_sequence(sorted_x, sorted_x_len.cpu(), batch_first=True)
        pack_output, _ = self.lstm(pack_input, None)
        x, _ = pad_packed_sequence(pack_output, batch_first=True)
        x = x[recover_idx, :]
        
        x = self.fc(x)

        logits = torch.sum(x, dim=1)
        
        return logits

        
