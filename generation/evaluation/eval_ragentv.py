import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
from tqdm import tqdm
import ast
import json
import pandas as pd
from utils import process_label, get_srls, update
from generation_utils import generate_step, generate_llm, VECTORSTORES, remove_duplicates
from refinement_utils import verify, get_refined_policy, get_refined_policy_llm
from prompts import get_error_instrution, ID2AUGS
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, set_seed
from peft import PeftModel
from evaluator import AccessEvaluator
from pathlib import Path
import click
import time


MAX_TRIES = 10
    

def generate_refine(mode, df, gen_model, gen_tokenizer, ver_model, ver_tokenizer, refine=True, save_generations=True, use_pipe=False, no_retrieve=False, no_update=False, result_dir = "results/eval", n=5):
    postfix = ""
    
    if refine:
        postfix+="_refined"
    if no_retrieve:
        postfix+="_no_retrieve"
    if no_update:
        postfix+="_no_update"
    
    
    inputs, predpols, truepols, predsrls, truesrls = [],[],[],[],[]
    generations = []
    id = 0
    fails = 0
    
    if use_pipe:
        gen_pipe = pipeline(
            "text-generation",
            model=gen_model,
            tokenizer=gen_tokenizer,
            device='cuda:0'
        )
    
    for nlacp, truth, origin in tqdm(zip(df['input'].to_list(), df['output'].to_list(), df['origin'].to_list())):
        
        max_tries = 0
        
        id+=1
        
        t,_ = process_label([truth])
        # t = ast.literal_eval(str(truth))
        
        if use_pipe:
            p, store = generate_step(nlacp, origin, gen_pipe, gen_tokenizer, no_retrieve, n = n)
        else:
            p, store = generate_llm(nlacp, origin, gen_tokenizer, gen_model, no_retrieve, n = n)
        
        generations.append({
            'nlacp': nlacp,
            # 'ents': ents,
            'truth': t,
            'origin': origin,
            'policy': p
        })
        
        true_srl = get_srls(t)
        
        if not no_retrieve:
            if no_update:
                pred_pol = p
                pred_srl = get_srls(p, post_process=True)
            else:
                pred_pol, pred_srl = update(p, VECTORSTORES[store])
        else:
            refine = False
            pred_pol = p
            pred_srl = get_srls(p, post_process=True)
            
        
        if refine:
            status = [
                {
                    'nlacp': nlacp,
                    'truth': t
                }
            ]
            
            if p == []:
                error_class = 10
            else:
                error_class = verify(nlacp, pred_pol, ver_model, ver_tokenizer)
            
            status.append({
                'iteration': max_tries,
                'pred': pred_pol,
                'verification': ID2AUGS[error_class]
            })
            
            while error_class != 11 and max_tries < MAX_TRIES:
            
                try:
                    max_tries += 1
                    error_instruction = get_error_instrution(nlacp, pred_pol, error_class)
                    
                    if use_pipe:
                        pred_t = get_refined_policy(error_instruction, gen_pipe, gen_tokenizer)
                    else:
                        pred_t = get_refined_policy_llm(error_instruction, gen_model, gen_tokenizer)
                        
                    pred_pol = ast.literal_eval(str(pred_t))
                    assert len(pred_pol) > 0, f'Re-generation failed for: \n{nlacp}\n'
                    pred_pol = remove_duplicates(pred_pol)
                    error_class = verify(nlacp, pred_pol, ver_model, ver_tokenizer)
                    
                    status.append({
                        'iteration': max_tries,
                        'pred': pred_pol,
                        'verification': ID2AUGS[error_class]
                    })
                    
                except:
                    fails+=1
                    print(f'Fail count: {fails}')
                    break
            
            pred_srl = get_srls(pred_pol)
            
            with open(f'{result_dir}/{mode}/refinements/{id}.json', 'w') as f:
                json.dump(status, f)

        inputs.append(nlacp)
        predpols.append(pred_pol)
        truepols.append(t)
        predsrls.append(pred_srl)
        truesrls.append(true_srl)
        
    if save_generations:
        with open(f'{result_dir}/{mode}/generations{postfix}.json', 'w') as f:
            json.dump(generations, f)
        
    e = AccessEvaluator(truepols, predpols, truesrls, predsrls, inputs, None, None, cats=None)
    ress,_ = e.evaluate()
    
    return ress

@click.command()
@click.option('--mode', default='ibm', 
              help='Mode of training (document-fold you want to evaluate the trained model on)',
              show_default=True,
              required=True,
              type=click.Choice(['t2p', 'acre', 'ibm', 'collected', 'cyber','overall'], case_sensitive=False))
@click.option('--result_dir', default='results/sarcp/', 
              help='Directory to save evaluation results',
              show_default=True,
              )
@click.option('--n', default=5, 
              help='Number of entities to retrieve per each component',
              show_default=True,
              )
@click.option('--refine', is_flag=True)
@click.option('--no_retrieve', is_flag=True)
@click.option('--no_update', is_flag=True)
@click.option('--use_pipe', is_flag=True)
def main(mode, result_dir, n, refine, no_retrieve, no_update, use_pipe):
    
    set_seed(1)
    
    results_dir = f"{result_dir}/{mode}/refinements"
    path = Path(results_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(f"../../data/document_folds/{mode}_acp.csv")
    
    if 'origin' not in df.columns:
        df['origin'] = [mode]*len(df)
    # df['origin'] = ['ibm']*len(df)
    
    GEN_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
    VER_MODEL = "facebook/bart-large"
    
    gen_ckpt = f"../../checkpoints/generation/{mode}/checkpoint/"
    ver_ckpt = "../../checkpoints/verification/checkpoint/"
    
    model_kwargs = dict(
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # loading the model with flash-attenstion support
        torch_dtype=torch.bfloat16
    )
    base_model = AutoModelForCausalLM.from_pretrained(GEN_MODEL, **model_kwargs)

    gen_model = PeftModel.from_pretrained(base_model, gen_ckpt).to('cuda:0')

    gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL, use_fast=True)
    gen_tokenizer.pad_token = gen_tokenizer.eos_token
    gen_tokenizer.padding_side = 'left'
    
    ver_model, ver_tokenizer = None, None
    
    if refine:
    
        ver_tokenizer = AutoTokenizer.from_pretrained(VER_MODEL)

        ver_model = AutoModelForSequenceClassification.from_pretrained(ver_ckpt).to(
            "cuda:0"
        )
        ver_model = ver_model.eval()
    
    start = time.time()
    results = generate_refine(
        mode, df, gen_model, gen_tokenizer, ver_model, ver_tokenizer, refine=refine, save_generations=True, use_pipe=use_pipe, no_retrieve=no_retrieve, no_update=no_update,
        result_dir = result_dir, n=n
    )
    end = time.time()
    
    print("\n\n ========================= Results (SARCP) ================================= \n")
    print(f'Mode: {mode}\nRefinement: {refine}\nRetrieval" {not no_retrieve}\nPost-process: {not no_update}\nUsing pipe: {use_pipe}\nTime elapsed: {end-start}s\n')
    print(f"Precision: {results['srl']['precision']}\nRecall: {results['srl']['recall']}\nF1: {results['srl']['f1']}\n")
    print(f"Component-wise results: {results['srl']['components']}\n")
    
    print("\n ========================= Results (ACR Generation) ========================= \n")
    
    print(f"Precision: {results['ent_type']['precision']}\nRecall: {results['ent_type']['recall']}\nF1: {results['ent_type']['f1-score']}\n")
    print(" ============================================================================ \n\n")

                
if __name__ == '__main__':
    main()
    
        
        


        
    

        
         
    