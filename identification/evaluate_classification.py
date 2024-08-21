import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import click
from transformers import EvalPrediction
import numpy as np
from metrics import compute_metrics
import pandas as pd
from classification_dataset import ACPDataset

from transformers import BertForSequenceClassification, BertTokenizerFast
from torch.utils.data import DataLoader

def evaluate(loader, model, device):
    
    all_preds = None
    all_labels = None

    for i,batch in enumerate(loader):
        input = {k: v.to(device) for k,v in batch.items()}

        out = model(**input)
        
        logits = out.logits.cpu().detach().numpy()
        labels = input['labels'].cpu().detach().numpy()
        
        all_preds = logits if all_preds is None else np.concatenate((all_preds, logits))
        all_labels = labels if all_labels is None else np.concatenate((all_labels, labels))
        
    eval_pred = EvalPrediction(predictions = all_preds, label_ids=all_labels)
    return compute_metrics(eval_pred)

@click.command()
@click.option('--mode', default='t2p', 
              help='Mode of training (document-fold you want to evaluate the trained model on)',
              show_default=True,
              required=True,
              type=click.Choice(['t2p', 'acre', 'ibm', 'collected', 'cyber', 'overall'], case_sensitive=False)
              )
def main(mode):
    
    """ Evaluates the NLACP identification module."""
    
    NUM_CLASSES = 2
    
    test_path = f'../data/document_folds/{mode}.csv'
        
    checkpoint = f'../checkpoints/identification/{mode}/checkpoint'

    model = BertForSequenceClassification.from_pretrained(checkpoint, num_labels=NUM_CLASSES).to('cuda:0')
    tokenizer = BertTokenizerFast.from_pretrained(checkpoint)

    
    test_df = pd.read_csv(test_path)
    test_ds = ACPDataset(test_df, tokenizer)
    test_dataloader = DataLoader(test_ds, num_workers=1, batch_size=16)
    
    print('\n =============================== Evaluating ================================ \n')
    
    
    results = evaluate(test_dataloader, model, 'cuda:0')
    
    print(f'Test path: {test_path}\n')
    print(results)
    
    print('\n =========================================================================== \n')
    
    
if __name__ == '__main__':
    main()
    
    