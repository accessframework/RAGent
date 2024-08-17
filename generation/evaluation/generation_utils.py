import re
import ast
from utils import get_available_entities,load_vectorstores
from prompts import get_generation_msgs, get_generation_msgs_ents

VECTORSTORES = load_vectorstores()

def generate_llm(nlacp, origin, tokenizer, model, no_retrieve=False, setting="DSARCP"):
    
    terminators = [
        60, # ]
        933, # ]\n
        14711,
        22414, # \n\n\n\n
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    if origin.endswith('_train'):
        origin = 'misc'
        
    if not no_retrieve:
    
        ents = get_available_entities(nlacp, origin, VECTORSTORES)

        messages = get_generation_msgs_ents(nlacp, ents)
    else:
        messages = get_generation_msgs(nlacp)
        
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    outputs = model.generate(
        input_ids,
        max_new_tokens=1024,
        eos_token_id=terminators,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False
    )
        
    response = outputs[0][input_ids.shape[-1]:]
    resp = tokenizer.decode(response, skip_special_tokens=True).strip()
    try:
        
        pattern = r'\[.*?\]'
        str_policy = re.findall(pattern, resp)[0]
        p = ast.literal_eval(str_policy)
        
        if setting.lower() == 'sar':
            sar_policy = []
            for rule in p:
                sar_p = {'subject': 'none', 'action': 'none', 'resource': 'none'}
                
                for k,v in rule.items():
                    if k in sar_p:
                        sar_p[k] = v
                if sar_p not in sar_policy:
                    sar_policy.append(sar_p)
            return sar_policy, origin
        return p, origin
    except:
        print('-'*30)
        print(nlacp)
        print(resp)
        print('-'*30)
        return [], origin

def generate_step(nlacp, origin, pipe, tokenizer, no_retrieve=False, setting = 'DSARCP'):

    terminators = [
        60, # ]
        933, # ]\n
        14711,
        22414, # \n\n\n\n
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    generation_args = {
        "max_new_tokens": 1024,
        "return_full_text": False,
        "do_sample": False,
        "eos_token_id": terminators,
    }
    
    if origin.endswith('_train'):
        origin = 'misc'
        
    if not no_retrieve:
    
        ents = get_available_entities(nlacp, origin, VECTORSTORES)

        messages = get_generation_msgs_ents(nlacp, ents)
    else:
        messages = get_generation_msgs(nlacp)
        
    response = pipe(messages, **generation_args)
    resp = response[0]['generated_text'].strip()
    try:
        
        pattern = r'\[.*?\]'
        str_policy = re.findall(pattern, resp)[0]
        p = ast.literal_eval(str_policy)
        if setting.lower() == 'sar':
            sar_policy = []
            for rule in p:
                sar_p = {'subject': 'none', 'action': 'none', 'resource': 'none'}
                
                for k,v in rule.items():
                    if k in sar_p:
                        sar_p[k] = v
                if sar_p not in sar_policy:
                    sar_policy.append(sar_p)
            return sar_policy, origin
        return p, origin
    except:
        print('-'*30)
        print(nlacp)
        print(resp)
        print('-'*30)
        return [], origin
    
def remove_duplicates(policy):
    
    uniques = []
    
    for r in policy:
        if r not in uniques:
            uniques.append(r)
            
    return uniques
