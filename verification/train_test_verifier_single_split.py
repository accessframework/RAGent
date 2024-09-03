import os
import torch
from transformers import (
    AutoTokenizer,
    Trainer,
    AutoTokenizer,
    TrainingArguments,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    set_seed
)
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, matthews_corrcoef
import json
import numpy as np
import pandas as pd
import click
from verification_dataset import VerificationDatasetSent
from utils import compute_metrics, preprocess_logits_for_metrics_bart, prepare_inputs_bart
from pathlib import Path


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


    print(
        classification_report(truth_bin, preds_bin, target_names=["negative", "positive"])
    )
    print()
    print(f"MCC: {matthews_corrcoef(np.array(truth_bin), np.array(preds_bin))}")
    print()
    print(classification_report(truth, preds))
    print("=="*50)
    print()
    
    del ver_model
    


def train_test_verifier(fold, id2augs, batch_size = 16, learning_rate = 2e-5, train_epochs =10, ckpt_dir = 'checkpoints/bart'):
    
    set_seed(1)
    model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large', num_labels = len(id2augs))
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large')
        
    train_ds = VerificationDatasetSent(fold['train'], tokenizer)
    val_ds = VerificationDatasetSent(fold['val'], tokenizer)
    test_ds = VerificationDatasetSent(fold['test'], tokenizer)
    
    path = Path(ckpt_dir)
    path.mkdir(parents=True, exist_ok=True)

    
    training_args = TrainingArguments(
        output_dir=ckpt_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_steps= 100,
        evaluation_strategy='steps',
        num_train_epochs=train_epochs,
        weight_decay=0.01,
        save_strategy="steps",
        save_steps=100,
        logging_steps=100,
        report_to='none',
        do_eval=True,
        lr_scheduler_type='cosine',
        warmup_ratio=0.03,
        load_best_model_at_end=True,
        save_total_limit=1,
        metric_for_best_model='eval_f1-macro',
        greater_is_better=True
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics_bart,
    )

    trainer.train()
    
    testing_results = trainer.evaluate(test_ds, metric_key_prefix='test')
    
    ckpts = [name for name in os.listdir(ckpt_dir)]
    
    del model
    eval_ckpt(fold['test'], ckpt_dir+"/"+ckpts[0])
    
    return testing_results


def train_verifier(ds_path, id2augs, batch_size = 16, learning_rate = 2e-5, 
                   train_epochs =10, ckpt_dir = 'checkpoints/bart'
                   ):
    
    TRAIN_DF = pd.read_csv(f'{ds_path}/utrain.csv')
    TEST_DF = pd.read_csv(f'{ds_path}/utest.csv')
    VAL_DF = pd.read_csv(f'{ds_path}/uval.csv')
    
    fold = {
        'train': TRAIN_DF,
        'val': VAL_DF,
        'test': TEST_DF
    }
    
        
    test_results = train_test_verifier(fold, id2augs = id2augs, batch_size = batch_size, learning_rate = learning_rate, train_epochs =train_epochs, ckpt_dir=ckpt_dir)
        
    with open('results.json','w') as f:
        json.dump(test_results, f)
    

    

@click.command()
@click.option('--base_model', 
       help='Base model',
       type=click.Choice(['bart', 'deberta'], case_sensitive=False),
       default='bart',
       show_default=True
       )
@click.option('--dataset_path', 
              help='Directory generated verification datasets',
              default="../data/verification",
              show_default=True,
              required=True)
@click.option('--train_epochs', default=10, help='Number of epochs to train',show_default=True)
@click.option('--learning_rate', default=2e-5, help='Learning rate',show_default=True)
@click.option('--batch_size', default=8, help='Batch size',show_default=True)
@click.option('--out_dir', default='../checkpoints/verification', help='Output directory',show_default=True)
def main(base_model, dataset_path, batch_size = 16, learning_rate = 2e-5, train_epochs = 10, 
                   out_dir = 'checkpoints/bart_bin'):
    
    """Trains the access control policy verifier using a single random train, val, test splits"""
    
    ID2AUGS = {0: 'allow_deny',
                1: 'csub',
                2: 'cact',
                3: 'cres',
                4: 'ccond',
                5: 'cpur',
                6: 'msub',
                7: 'mres',
                8: 'mcond',
                9: 'mpur',
                10: 'mrules',
                11: 'correct'}
    
    print('\n =========================== Training details =========================== \n')
    print(f'Base model: {base_model}\nDataset: {dataset_path}\nNum. of classes: {len(ID2AUGS)}\nNum. of epochs: {train_epochs}\nLearning rate: {learning_rate}\nBatch size: {batch_size}\nCheckpoint dir.: {out_dir}\n')
    print(' ======================================================================= \n')

    train_verifier(ds_path=dataset_path, id2augs=ID2AUGS, batch_size = batch_size, learning_rate = learning_rate, train_epochs=train_epochs, ckpt_dir = out_dir)
    

if __name__ == '__main__':
    main()