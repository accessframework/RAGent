import click
from torchtune.datasets._alpaca import AlpacaInstructTemplate

from datasets import load_dataset
from peft import LoraConfig
import torch
from pprint import pprint
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, set_seed

MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    
    return tokenizer
    

def train(train_set, validation_set, training_config, peft_config):
    
    model_kwargs = dict(
        use_cache=False,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # loading the model with flash-attenstion support
        torch_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(MODEL, **model_kwargs)
    
    tokenizer = get_tokenizer()
    
    train_conf = TrainingArguments(**training_config)
    peft_conf = LoraConfig(**peft_config)
    
    trainer = SFTTrainer(
        model=model,
        args=train_conf,
        peft_config=peft_conf,
        train_dataset=train_set,
        eval_dataset=validation_set,
        max_seq_length=2048,
        dataset_text_field="text",
        tokenizer=tokenizer,
        packing=False
    )
    train_result = trainer.train()
    
    return train_result

def apply_alpaca_template(example, tokenizer):

    output = example['output']
    example['text'] = AlpacaInstructTemplate.format(example) + output + "\n" + tokenizer.eos_token
    return example

def apply_chat_template(example, tokenizer):
    messages = example["messages"]
    # We add an empty system message if there is none
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": ""})
    example["text"] = tokenizer.apply_chat_template(messages, tokenize=False)

    return example

def get_datasets(path, test_size = 0.2, seed=1):

    dataset = load_dataset(path, split='train')
    splits = dataset.train_test_split(test_size = test_size, seed = seed)
    train_dataset, test_dataset = splits['train'], splits['test']
    column_names = list(train_dataset.features)
    tokenizer = get_tokenizer()
    
    processed_train_dataset = train_dataset.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=10,
        remove_columns=column_names,
        desc="Applying chat template to train_sft",
    )

    processed_test_dataset = test_dataset.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=10,
        remove_columns=column_names,
        desc="Applying chat template to test_sft",
    )
    
    print(f"Example train: {processed_train_dataset[11]}\n")
    print(f"Example validation: {processed_test_dataset[11]}\n")
    
    return processed_train_dataset, processed_test_dataset


@click.command()
@click.option('--train_path', 
       help='Huggingface dataset name',
       required=True
       )
@click.option('--out_dir', default='../checkpoints/', help='Output directory',show_default=True)
@click.option('--batch_size', default=8, help='Batch size',show_default=True)
@click.option('--lr', default=2e-4, help='Learning rate',show_default=True)
@click.option('--seed', default=1, help='Random seed',show_default=True)
def main(train_path, batch_size, lr, out_dir, random_seed):
    
    """Training the LLaMa 3 8B for generating access control policies from NLACPs"""
    
    set_seed(seed=random_seed)
    train_set, validation_set = get_datasets(train_path, test_size = 0.2, seed=random_seed)
    
    training_config = {
        "bf16": True,
        "do_eval": True,
        "learning_rate": lr,
        "logging_steps": 100,
        "eval_steps": 100,
        "logging_strategy": "steps",
        "evaluation_strategy": "steps",
        "lr_scheduler_type": "cosine",
        "max_steps": -1,
        "num_train_epochs": 2,
        "output_dir": out_dir,
        "overwrite_output_dir": True,
        "per_device_eval_batch_size": batch_size,
        "per_device_train_batch_size": batch_size,
        "remove_unused_columns": True,
        "save_steps": 100,
        "optim": "adamw_8bit",
        "seed": random_seed,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs":{"use_reentrant": False},
        "gradient_accumulation_steps": 1,
        "warmup_ratio": 0.2,
        "report_to": "none"
        }

    peft_config = {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": "all-linear"
    }
    
    print("\n\n=================================== Configs ====================================\n\n")
    print("\n\n------------------------------- Training configs -------------------------------\n")
    pprint(training_config)
    print("\n--------------------------------- PEFT configs ---------------------------------\n")
    pprint(peft_config)
    print()
    
    train(train_set, validation_set, training_config, peft_config)
    
if __name__ == '__main__':
    main()
    
    


