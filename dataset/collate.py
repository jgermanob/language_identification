import torch

def collate_fn(instances):
    batch_size = len(instances)
    batch_data = sorted(instances, key=lambda instance: len(instance.input.chars), reverse=True)
    char_seq_len = torch.LongTensor(list(map(lambda instance: len(instance.input.chars), batch_data)))
    max_char_seq_len = char_seq_len.max()
    label_len = len(instances[0].output)
    
    # Create tensors
    batch_char_seq_tensor = torch.zeros((batch_size, max_char_seq_len), dtype=torch.long)
    batch_labels_tensor = torch.zeros((batch_size, label_len), dtype=torch.float)

    for idx in range(batch_size):
        batch_char_seq_tensor[idx, :char_seq_len[idx]] = torch.LongTensor(batch_data[idx].input.chars)
        batch_labels_tensor[idx, :label_len] = torch.LongTensor(batch_data[idx].output)

    batch = {'x_chars': batch_char_seq_tensor,
             'x_chars_len': char_seq_len,
             'labels': batch_labels_tensor}
    return batch    
