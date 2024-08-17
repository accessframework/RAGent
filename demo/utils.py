import torch
import ast
import random
import ast
import re
import json
from enum import Enum


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

class Error(Enum):
    MISSING_COMPONENT = 'missing_component'
    INCORRECT_COMPONENT = 'incorrect_component'
    CHANGED_DECISION = 'changed_decision'
    MISSING_RULES = 'missing_rules'

ERROR_MAP = {
    0:(Error.CHANGED_DECISION, ''), 
    1:(Error.INCORRECT_COMPONENT, 'subject'), 
    2:(Error.INCORRECT_COMPONENT, 'action'), 
    3:(Error.INCORRECT_COMPONENT, 'resource'), 
    4:(Error.INCORRECT_COMPONENT, 'condition'), 
    5:(Error.INCORRECT_COMPONENT, 'purpose'), 
    6:(Error.MISSING_COMPONENT, 'subject'), 
    7:(Error.MISSING_COMPONENT, 'resource'), 
    8:(Error.MISSING_COMPONENT, 'condition'), 
    9:(Error.MISSING_COMPONENT, 'purpose'),
    10:(Error.MISSING_RULES, '')
}

ID2AUGS = {
    0: "allow_deny",
    1: "csub",
    2: 'cact',
    3: "cres",
    4: "ccond",
    5: "cpur",
    6: "msub",
    7: "mres",
    8: "mcond",
    9: "mpur",
    10: "mrules",
    11: "correct",
}

ACP_DEFN = """
Access Control Policy Definition
--------------------------------

Access Control Policy (ACP) is a python list containing python dictionaries.
Each python dictionary represents an Access Control Rule (ACR).
Each ACR (or python dictionary) contains five keys, namely, 'decision', 'subject', 'action', 'resource', 'purpose', and 'condition'.

'decision': The access decision of the ACR. It can only be either 'allow' used for an ACR that allows a subject to perform an action on a resource, or 'deny' otherwise.
'subject': The entity that performs the 'action' on the 'resource'. 
'resource': The entity which the 'subject' performs an 'action' on.
'purpose': The purpose of perfoming the 'action'.
'condition': The condition/constraint affect the 'action'.

It is important that each ACP can contain one or more ACRs and each ACR can contain only one subject, action, resource, purpose, or condition.
When generating the ACP, first you need to identify ACRs.
For example, in the sentence 'The doctor and the nurse can read records if the patient agrees.' contains two ACRs: 'The doctor can read records if the patient agrees' and 'The nurse can read records if the patient agrees'.
Then represent each identified ACR as a python dictionary.
For example, the ACR 'The doctor can read records if the patient agrees.' has the 'allow' access decision since the ACR allows the doctor to read records, 'subject' is the doctor, the resource is the 'record', and the 'condition' is 'the patient agrees', since the patient's agreement affect the action 'read'.
Here there is no 'purpose' mentioned in the ACR. Therefore the value for the key 'purpose' is 'none'.
Finally we have an ACR as, 
{'decision': 'allow', 'subject': 'doctor', 'action': 'read', 'resource': 'record', 'purpose': 'none', 'condition': 'patient agrees'}.
Similary, for the ACR 'The nurse can read records if the patient agrees', we can build the python dictionary as,
{'decision': 'allow', 'subject': 'nurse', 'action': 'read', 'resource': 'record', 'purpose': 'none', 'condition': 'patient agrees'}.
Finally the ACP can be created as a python list with built ACR dictionaries.

ACP: [{'decision': 'allow', 'subject': 'doctor', 'action': 'read', 'resource': 'record', 'purpose': 'none', 'condition': 'patient agrees'}, {'decision': 'allow', 'subject': 'nurse', 'action': 'read', 'resource': 'record', 'purpose': 'none', 'condition': 'patient agrees'}]
\n\n
"""

INSTRUCTION_ENTS = f"{ACP_DEFN}\nGiven a natural language sentence (i.e., NLACP), generate an access control policy according to the above Access Control Policy Definition.\nIf a value for any key in any python dictionary is cannot be found in the NLACP, set the value to 'none'.\nTo identify subject, resource, purpose, and condition, use the entities provided as a dictionary, 'Available entities' as an aid.\nIf none of the provided Available entities match the entities of the NLACP or there is no 'Available entities' provided, use your judgment to select the most suitable entity within the NLACP\nDo not use values in a different list as values for the keys subject, resource, purpose, and condition of the dictioaries represent access control rules (i.e., Do not use values available for 'resource' as 'subject' in in the access control policy, if it is not listed as an available 'subject').\n"

basic_template = "{subject} is {decision} {action} {resource}"
basic_template_subject = "{subject} is {decision} {action}"
basic_template_resource = "{resource} is {decision} be {action}"

purpose_template = "{template} for the purpose of {purpose}"
condition_template = "{template} if {condition}"
sarcp_template = "{template} for the purpose of {purpose}, if {condition}"

allow_sents = ['allowed to', 'able to', 'shall', 'authorized to']
deny_sents = ['prohibited to', 'not able to', 'shall not', 'unauthorized to']


def get_generation_msgs_ents(nlacp, with_ents = True, ent_file=None):
    if ent_file == None:
        # ents = {'subject': ["hotcrp.com", "site's manager", "site manager",'program chair', "request", "user", 'you'], 'resource': ['artifact', 'data', 'submission artifact', 'review artifact', 'profile data', 'demographic data', 'browsing data', 'information', 'configuration setting', 'profile'], 'purpose': ['review or other purposes'], 'condition': ['artifact was submitted in error', 'allowed by Site managers site agreements with HotCRP.com', 'upon submitting an artifact to a HotCRP.com site']}
        with_ents = False
    else:
        with open(ent_file, 'r') as f:
            ents = json.load(f)
            
    if with_ents:
        user_msg = f"NLACP: {nlacp}\nAvailable entities: {ents}"
    else:
        user_msg = f"NLACP: {nlacp}"
    
    messages = [
        {"role": "system", "content": INSTRUCTION_ENTS},
        {"role": "user", "content": user_msg},
    ]
    return messages

def get_error_instrution(nlacp, gen_t_1, error_class):
    
    error = ERROR_MAP[error_class]
    error_prompt = get_error_prompt(error)
    return ACP_DEFN + f"You generated \'{gen_t_1}\' for the sentence, \'{nlacp} based on the mentioned Access Control Policy Definition.\nHowever the follwoing error is found.\n\n1. {error_prompt}\n\nPlease address the error and output the corrected access control policy according to the mentioned definition.\nThink step-by-step to first provide the reasoning steps/thought process.\nThen after the \n\n### Corrected: \n\n provide the corrected policy without any other text. Then add \n\n####"


def get_error_prompt(error):

    category,component = error

    if category == Error.MISSING_COMPONENT:
        return f"According to the Access Control Policy Definition, the value assigned for the '{component}' in one or more access control rules are 'none' (indicating not available in the sentence), even though the value can be found\n"

    elif category == Error.INCORRECT_COMPONENT:
        return f"According to the Access Control Policy Definition, the value assigned for the '{component}' in one or more access control rules are incorrect\n"

    elif category == Error.CHANGED_DECISION:
        return f"According to the Access Control Policy Definition, the value assigned for the decision in one or more access control rules are incorrect\n"

    else:
        return f"According to the Access Control Policy Definition, one or more python dicionaries that represent access control rules related to actions are missing\n"

def convert_to_sent(sample):
    error = 0
    sample = ast.literal_eval(str(sample))
    pol_sent = []
    for rule in  sample:

        if rule['decision'] == 'allow':
            rand = random.randint(0, len(allow_sents)-1)
            decision = allow_sents[rand]
        else:
            rand = random.randint(0, len(deny_sents)-1)
            decision = deny_sents[rand]
    
        subject = rule['subject']
        action = rule['action']
        resource = rule['resource']
        purpose = rule['purpose']
        condition = rule['condition']
    
        if subject != 'none' and resource != 'none':
            template = basic_template.format_map({'decision': decision, 'subject': subject, 'action': action, 'resource': resource})
        elif subject != "none":
            template = basic_template_subject.format_map({'decision': decision, 'subject': subject, 'action': action})
        elif resource != "none":
            template = basic_template_resource.format_map({'decision': decision, 'resource': resource, 'action': action})
    
        else:
            # print(rule)
            template = ''
            error = True
    
    
        if purpose != 'none' and condition != 'none':
    
            final_template = sarcp_template.format_map({'template': template, 'purpose': purpose, 'condition': condition})
        elif purpose != 'none':
            final_template = purpose_template.format_map({'template': template, 'purpose': purpose})
    
        elif condition != 'none':
            final_template = condition_template.format_map({'template': template, 'condition': condition})
        else:
            final_template = template

        pol_sent.append(final_template)
    return " and ".join(pol_sent).strip(), error


def postprocess(key, val):
    
    l = ['any', 'all', 'every','the']
    conds = ['when', 'if', 'if,']
    purs = ['for', 'to']
    
    val = val.replace('_',',').replace('-', '').replace("'", "").replace('’', '').replace("”", "").replace("“","")
    
    if key == 'subject' or key == 'resource':
        for k in l:
            if val.split(' ')[0] == k:
                val = ' '.join(val.split(' ')[1:])
            
    if key=='purpose' and val.split(' ')[0] in purs:
        val = ' '.join(val.split(' ')[1:])
        
    if key=='condition' and val.split(' ')[0] in conds:
        val = ' '.join(val.split(' ')[1:])
    
    if val[-1] == 's' and val[-2] != 's':
        val = val[:-1]
        
    return ' '.join(val.split())

def get_srls(policy, post_process = True):
    
    srls = {}
    policies = []
    if post_process:
        
        for p in policy:
            rule = {'decision': 'allow', 'subject': 'none', 'action': 'none', 'resource': 'none', 'condition': 'none', 'purpose': 'none'}
            for key, val in p.items():
                key = key.strip()
                if key in rule:
                    nval = postprocess(key, val.strip())
                    rule[key] = nval
            policies.append(rule)
    else:
        policies = policy
        
    
    for pp in policies:
        action = pp['action']
        if action not in srls:
            srls[action] = {'subject': [], 'resource': [], 'purpose': [], 'condition': []}
            if pp['subject']!='none':
                srls[action]['subject'].append(pp['subject'])
            if pp['resource']!='none':
                srls[action]['resource'].append(pp['resource'])
            if pp['purpose']!='none':
                srls[action]['purpose'].append(pp['purpose'])
            if pp['condition']!='none':
                srls[action]['condition'].append(pp['condition'])    
        else:
            if pp['subject']!='none':
                srls[action]['subject'].append(pp['subject'])
            if pp['resource']!='none':
                srls[action]['resource'].append(pp['resource'])
            if pp['purpose']!='none':
                srls[action]['purpose'].append(pp['purpose'])
            if pp['condition']!='none':
                srls[action]['condition'].append(pp['condition'])
                
    return srls
    
def prepare_inputs_bart(s,l,tokenizer, device = 'cuda:0'):
    
    tokens = tokenizer(s,l,return_tensors='pt')
    
    return {k:v.to(device) for k,v in tokens.items()}


def generate(nlacp, tokenizer, model, with_ents = True, ent_file=None):
    
    terminators = [
        60, # ]
        933, # ]\n
        14711,
        22414, # \n\n\n\n
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    messages = get_generation_msgs_ents(nlacp, with_ents, ent_file)
        
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
        return p
    except:
        print('-'*30)
        print(nlacp)
        print(resp)
        print('-'*30)
        return []

def remove_duplicates(policy):
    
    uniques = []
    
    for r in policy:
        if r not in uniques:
            uniques.append(r)
            
    return uniques

def prepare_inputs_bart(s,l,tokenizer, device = 'cuda:0'):
    
    tokens = tokenizer(s,l,return_tensors='pt')
    
    return {k:v.to(device) for k,v in tokens.items()}



def verify(nlacp, pred_pol, ver_model, ver_tokenizer):
    # print(pred_pol)
    try:
        pp, _ = convert_to_sent(str(pred_pol))
        
        ver_inp = prepare_inputs_bart(nlacp, pp, ver_tokenizer, "cuda:0")
        pred = ver_model(**ver_inp).logits
    
        probs = torch.softmax(pred, dim=1)
        mprob = probs.max().item()
        pred_class = probs.argmax(axis=-1).item()
        highest_prob_cat = ID2AUGS[pred_class]
        
        return pred_class, mprob
    except:
        return 10, 1.0
    

def get_refined_policy_llm(instruction, model, tokenizer):
    terminators_errors = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    error_messages = [
        {'role': 'system', 'content': ''},
        {'role': 'user', 'content': instruction}
    ]

    input_ids = tokenizer.apply_chat_template(
        error_messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    outputs = model.generate(
        input_ids,
        max_new_tokens=1024,
        eos_token_id=terminators_errors,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False
    )
        
    response = outputs[0][input_ids.shape[-1]:]
    resp = tokenizer.decode(response, skip_special_tokens=True).strip()
    
    if ("### Corrected" in resp):

        resp = resp.split('### Corrected')[1]
    else:

        resp = resp.split('final refined policy')[1]

    pattern = r'\[.*?\]'
    sanitized = re.findall(pattern, resp)
    
    return sanitized[0]



def is_nlacp(input, id_model, id_tokenizer):
    
    input = input.replace('\u2019s',"")
    tokens = id_tokenizer(input, return_token_type_ids = False, return_tensors="pt").to('cuda:0')

    out = id_model(**tokens)
    res = torch.argmax(out.logits, dim=1)
    
    return res
    


def generate_policy(id, input, gen_model, gen_tokenizer, ver_model, ver_tokenizer, with_ents = True, ent_file=None):

    input = input.replace('\u2019s',"")
    
    MAX_TRIES = 10
    mprob = 1.0
    max_tries = 0
    status = []
    pred_pol = generate(input, gen_tokenizer, gen_model, with_ents=with_ents, ent_file=ent_file)

    if pred_pol == []:
        error_class = 10
    else:
        error_class,mprob = verify(input, pred_pol, ver_model, ver_tokenizer)

    
    status.append({
            'nlacp': input
        })
    status.append({
            'iteration': max_tries,
            'pred': pred_pol,
            'verification': ID2AUGS[error_class]
        })

    while error_class != 11 and max_tries < MAX_TRIES:
        
        try:
            max_tries += 1
            error_instruction = get_error_instrution(input, pred_pol, error_class)
            
            pred_t = get_refined_policy_llm(error_instruction, gen_model, gen_tokenizer)
            pred_pol = ast.literal_eval(str(pred_t))
            assert len(pred_pol) > 0, f'Re-generation failed for: \n{input}\n'
            pred_pol = remove_duplicates(pred_pol)
            error_class,mprob = verify(input, pred_pol, ver_model, ver_tokenizer)
            
            status.append({
                'iteration': max_tries,
                'pred': pred_pol,
                'verification': ID2AUGS[error_class]
            })
            
        except:
            # fails+=1
            print(pred_t)
            break

    if len(status)>2 :
        with open(f'generations/{id}.json', 'w') as f:
            json.dump(status, f)
            
    return pred_pol, error_class, mprob
    # else:
    #     return "Not an ACP"


