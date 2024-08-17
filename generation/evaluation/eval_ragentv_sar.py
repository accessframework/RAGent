import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
from tqdm import tqdm
import ast
import json
import pandas as pd
from utils import process_label, get_available_entities, load_vectorstores, convert_to_sent, prepare_inputs_bart, get_srls, update
from generation_utils import generate_step, generate_llm
import re
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import PeftModel
from pathlib import Path
from srl_results_sar import SRLEvalSAR
import click
CACHE = "/data/sjay950/huggingface/"

VECTORSTORES = load_vectorstores()
MAX_TRIES = 10
    

def generate_refine(mode, df, gen_model, gen_tokenizer, save_generations=True, use_pipe=False, result_dir='results/eval/sar/'):
    postfix = ""
    inputs, predpols, truepols, predsrls, truesrls = [],[],[],[],[]
    generations = []
    id = 0
    
    if use_pipe:
        gen_pipe = pipeline(
            "text-generation",
            model=gen_model,
            tokenizer=gen_tokenizer,
            device='cuda:0'
        )
    
    
    for nlacp, truth, origin in tqdm(zip(df['input'].to_list(), df['output'].to_list(), df['origin'].to_list())):
        
        id+=1
        
        t,_,true_srl = process_label([truth], 'sar')
        # t = ast.literal_eval(str(truth))
        
        if use_pipe:
            p, store = generate_step(nlacp, origin, gen_pipe, gen_tokenizer,setting='SAR')
        else:
            p, store = generate_llm(nlacp, origin, gen_tokenizer, gen_model, setting='SAR')
        
        generations.append({
            'nlacp': nlacp,
            # 'ents': ents,
            'truth': t,
            'origin': origin,
            'policy': p
        })
        
        pred_pol, pred_srl = update(p, VECTORSTORES[store])

        inputs.append(nlacp)
        predpols.append(pred_pol)
        truepols.append(t)
        predsrls.append(pred_srl)
        truesrls.append(true_srl)
        
    if save_generations:
        with open(f'{result_dir}{mode}/generations{postfix}.json', 'w') as f:
            json.dump(generations, f)
        
    true_srls, pred_srls = truesrls.copy(), predsrls.copy()

    
    with open(f'{result_dir}{mode}_pred_srl.json', '+w') as f:
        json.dump(predsrls, f)
        
    with open(f'{result_dir}{mode}_true_srl.json', '+w') as f:
        json.dump(truesrls, f)

    evaluator = SRLEvalSAR(true_srls, pred_srls)
    precision,recall,f1 = evaluator.get_f1()
        
    
    return precision,recall,f1

@click.command()
@click.option('--mode', default='ibm', 
              help='Mode of training (document-fold you want to evaluate the trained model on)',
              show_default=True,
              required=True,
              type=click.Choice(['t2p', 'acre', 'ibm', 'collected', 'cyber','overall'], case_sensitive=False))
@click.option('--result_dir', default='results/eval/sar/', 
              help='Directory to save evaluation results',
              show_default=True,
              )
@click.option('--use_pipe', is_flag=True)
def main(mode, result_dir, use_pipe):
    
    set_seed(1)
    
    results_dir = f"{result_dir}/{mode}"
    path = Path(results_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(f"../../data/document_folds/{mode}_acp.csv")
    
    if 'origin' not in df.columns:
        df['origin'] = [mode]*len(df)
    # df['origin'] = ['ibm']*len(df)
    
    GEN_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
    
    gen_ckpt = f"../../checkpoints/generation/{mode}/checkpoint/"
        
    model_kwargs = dict(
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # loading the model with flash-attenstion support
        torch_dtype=torch.bfloat16,
        cache_dir = CACHE
    )
    base_model = AutoModelForCausalLM.from_pretrained(GEN_MODEL, **model_kwargs)

    gen_model = PeftModel.from_pretrained(base_model, gen_ckpt).to('cuda:0')

    gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL, use_fast=True, cache_dir=CACHE)
    gen_tokenizer.pad_token = gen_tokenizer.eos_token
    gen_tokenizer.padding_side = 'left'
    
    precision,recall,f1 = generate_refine(
        mode, df, gen_model, gen_tokenizer, save_generations=True,use_pipe=use_pipe, result_dir=result_dir
    )
    
    print("\n==================== Results (SAR) ====================\n")
    print(f'Mode: {mode}\nUsing pipe: {use_pipe}')
    print(f'Precision: {precision}\nRecall: {recall}\nF1: {f1}\n')
    print("========================================================\n")

                
if __name__ == '__main__':
    main()
    
        
        


        
    

        
         
    