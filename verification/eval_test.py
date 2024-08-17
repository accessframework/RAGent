import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from sklearn.metrics import classification_report, matthews_corrcoef
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import click
from utils import prepare_inputs_bart


def eval_ckpt(df, ckpt):
    
    truth = df["labels"].to_list()
    truth_bin = []

    for i in truth:
        if i == 11:
            truth_bin.append(0)
        else:
            truth_bin.append(1)
            
    preds = []

    ver_tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large')

    ver_model = AutoModelForSequenceClassification.from_pretrained(ckpt).to(
        'cuda:0'
    ).eval()
    
    for i in tqdm(range(len(df))):

    
        input = df.iloc[i].inputs
        pp = df.iloc[i].policies_sent
            
        
        ver_inp = prepare_inputs_bart(input, pp, ver_tokenizer, 'cuda:0')
        pred = ver_model(**ver_inp).logits

        probs = torch.softmax(pred, dim=1)
        pred_class = probs.argmax(axis=-1).item()

        preds.append(pred_class)
    
    
    preds_bin = []

    for i in preds:
        if i == 11:
            preds_bin.append(0)
        else:
            preds_bin.append(1)


    print(f"ID: {id}\n")
    print(
        classification_report(truth_bin, preds_bin, target_names=["negative", "positive"])
    )
    print()
    print(f"MCC: {matthews_corrcoef(np.array(truth_bin), np.array(preds_bin))}")
    print()
    print(classification_report(truth, preds))
    print("=="*50)
    print()
    
@click.command()
@click.option('--ckpt', 
       help='Checkpoint location',
       default=9400,
       show_default=True
       )
def main(ckpt):

    df = pd.read_csv('../data/verification/verification_dataset_json_sent.csv')
    
    # Using the sane splits with random state used when training
    _, VAL_DF = train_test_split(df, test_size=0.2, random_state=1)
    VAL_DF, TEST_DF = train_test_split(VAL_DF, test_size=0.5, random_state=1)
    
    eval_ckpt(TEST_DF, ckpt)
    
    
if __name__ == '__main__':
    
    main()