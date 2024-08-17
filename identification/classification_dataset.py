import torch
from torch.utils.data import Dataset


class ACPDataset(Dataset):

    def __init__(self, df, tokenizer, max_len = 256):
        
        t = df['input'].values.tolist()
        self.text = [tokenizer(str(s), 
                        padding='max_length', 
                        max_length = max_len, 
                        truncation=True, 
                        return_token_type_ids = False, 
                        return_tensors="pt") for s in t]
        self.labels = torch.tensor(df['acp'].tolist())

    def __len__(self):
        assert len(self.text) == len(self.labels)
        return len(self.labels)

    def __getitem__(self, idx):
        output = {k: v.flatten() for k,v in self.text[idx].items()}
        output['labels'] = torch.tensor(int(self.labels[idx]))
        return output
        