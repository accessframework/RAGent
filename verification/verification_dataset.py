import torch
from torch.utils.data import Dataset
    
class VerificationDatasetSent(Dataset):
    
    def __init__(self, df, tokenizer, max_len = 1024):
        self.premise = df['inputs'].to_list()
        self.hypothesis = df['policies_sent'].to_list()
        self.labels = torch.tensor(df['labels'].to_list())
        
        self.input = [tokenizer(p,h,
                                return_tensors='pt',
                                padding='max_length',
                                truncation = True,
                                max_length = max_len) for p,h in zip(self.premise, self.hypothesis)]
        
          
    def __len__(self):
        return len(self.input)
    
    def __getitem__(self, index):
        output = {k: v.flatten() for k,v in self.input[index].items()}
        output['labels'] = self.labels[index]
        return output