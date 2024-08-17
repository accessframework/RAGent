import logging
import torch
from peft import PeftModel
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BertForSequenceClassification,
    BertTokenizerFast,
)
import pandas as pd
from tqdm import tqdm
import click
from utils import is_nlacp , generate_policy

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level="INFO",
)


def load_models_tokenizers(device):
    logging.info("Loading checkpoints ...")
    
    id_model_name = "../checkpoints/identification/overall/checkpoint/"

    id_model = BertForSequenceClassification.from_pretrained(
        id_model_name, num_labels=2
    ).to(device)
    id_tokenizer = BertTokenizerFast.from_pretrained(id_model_name)

    id_model.eval()

    ex_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    peft_model_id = "../checkpoints/generation/overall/checkpoint/"

    model_kwargs = dict(
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # loading the model with flash-attenstion support
        torch_dtype=torch.bfloat16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(ex_model_name, **model_kwargs)

    gen_model = PeftModel.from_pretrained(base_model, peft_model_id).to('cuda:0')

    gen_tokenizer = AutoTokenizer.from_pretrained(ex_model_name, use_fast=True)
    gen_tokenizer.pad_token = gen_tokenizer.eos_token
    gen_tokenizer.padding_side = 'left'
    
    gen_model.eval()

    ver_model_name = "facebook/bart-large"
    verification_ckpt = "../checkpoints/verification/checkpoint/"

    ver_tokenizer = AutoTokenizer.from_pretrained(ver_model_name)
    ver_model = AutoModelForSequenceClassification.from_pretrained(
        verification_ckpt
    ).to(device)
    ver_model.eval()

    return id_tokenizer, id_model, gen_tokenizer, gen_model, ver_tokenizer, ver_model


@click.command()
@click.option("--ent_file", default=None, help="JSON file containing predefined entities.", show_default=True)
def main(ent_file=None):
    
    id = 0

    logging.info("Loading preprocessed sentences ...")
    with open("high_level_requirements.json", "r") as f:

        sents = json.load(f)
    id_tokenizer, id_model, gen_tokenizer, gen_model, ver_tokenizer, ver_model = (
        load_models_tokenizers('cuda:0')
    )
    inputs, outputs = [], []
    
    if ent_file!=None:
        logging.info(f"Using the entity file: {ent_file}")
        with_ents = True
    else:
        logging.info("No entity file is provided.")
        with_ents = False

    logging.info("Generating ...")
    
    for s in tqdm(sents):
        inputs.append(s)
        nlacp = is_nlacp(s, id_model, id_tokenizer)
        if nlacp:
            output, _, _ = generate_policy(id, s, gen_model, gen_tokenizer, ver_model, ver_tokenizer, with_ents=with_ents, ent_file=ent_file)
        
        else:
            output = 'Not an ACP'

        outputs.append(output)
        id+=1

    df = pd.DataFrame(
        {"inputs": inputs, "outputs": outputs}
    )

    logging.info("The results are saved in: results.csv")

    df.to_csv("results.csv")

    print(
        "\n ================================ Completed ================================\n"
    )


if __name__ == "__main__":
    main()
