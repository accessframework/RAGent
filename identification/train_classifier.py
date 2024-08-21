import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd

from sklearn.model_selection import train_test_split
from pathlib import Path

from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import TrainingArguments, Trainer
from metrics import compute_metrics
from classification_dataset import ACPDataset
import click


def train_classifier(train_ds, val_ds, model, tokenizer, batch_size = 16, epochs = 10, learning_rate = 2e-5, out_dir = ""):
    
    training_args = TrainingArguments(
        output_dir=out_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        evaluation_strategy = 'epoch',
        weight_decay=0.01,
        save_strategy="epoch",
        # logging_steps=10,
        logging_strategy='epoch',
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to='none'
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    

@click.command()
@click.option('--dataset_path', help='Location of the dataset to train the model', required=True)
@click.option('--max_len', default=256, help='Maximum length for the input sequence')
@click.option('--batch_size', default=16, help='Batch size', required=True)
@click.option('--epochs', default=10, help='Number of epochs', required=True)
@click.option('--learning_rate', default=2e-5, help='Learning rate', required=True)
@click.option('--out_dir', default='../checkpoints/identification', help='Directory to save the checkpoints', required=True)
def main(dataset_path, max_len=256, batch_size = 16, epochs = 10, learning_rate = 2e-5, out_dir = ""):
    
    """Trains the NLACP identification module"""
    
    MODEL = 'bert-base-uncased'
    NUM_CLASSES = 2
    
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(dataset_path)
    
    train_df, val_df = train_test_split(df, test_size = 0.2, random_state=42)
    
    model = BertForSequenceClassification.from_pretrained(MODEL, num_labels=NUM_CLASSES)
    tokenizer = BertTokenizerFast.from_pretrained(MODEL)
    
    train_ds = ACPDataset(train_df, tokenizer, max_len=max_len)
    val_ds = ACPDataset(val_df, tokenizer, max_len=max_len)
    
    print('\n =========================== Training details =========================== \n')
    print(f'Dataset: {dataset_path}\nNum. of classes: {NUM_CLASSES}\nNum. of epochs: {epochs}\nLearning rate: {learning_rate}\nBatch size: {batch_size}\nCheckpoint dir.: {out_dir}\n')
    print(' ======================================================================= \n')
    
    train_classifier(train_ds, val_ds, model, tokenizer, batch_size = batch_size, epochs = epochs, learning_rate = learning_rate, out_dir = out_dir)
    

if __name__ == '__main__':
    main()