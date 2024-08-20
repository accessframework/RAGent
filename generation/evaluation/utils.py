import torch
import ast
import yaml
from pprint import pprint
import random
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


a,count = 0,0
failed_parse = []

basic_template = "{subject} is {decision} {action} {resource}"
basic_template_subject = "{subject} is {decision} {action}"
basic_template_resource = "{resource} is {decision} be {action}"

purpose_template = "{template} for the purpose of {purpose}"
condition_template = "{template} if {condition}"
sarcp_template = "{template} for the purpose of {purpose}, if {condition}"

allow_sents = ['allowed to', 'able to', 'shall', 'authorized to']
deny_sents = ['prohibited to', 'not able to', 'shall not', 'unauthorized to']


def create_out_string(inp):
    s = "{"
    # print(inp)
    for e in inp:
        # print(e)
        for k, v in e.items():
            s += f"{k}: {v}; "

        s = s[:-2] + " | "

    return s[:-3] + "}"

def format_labels(label: str):
    nlabel =  label.lower().replace('"', '').replace(';',',').replace('decision: ', "\'decision\': '").replace('subject: ', "\'subject\': '").replace('action: ', "\'action\': '").replace('resource: ', "\'resource\': '").replace('condition: ', "\'condition\': '").replace('purpose: ', "\'purpose\': '").replace(",", "',").replace('}', "'}").replace("'s ", "\\'s ")
    i = nlabel.find('decision')
    return ("{'" + nlabel[i:])

def postprocess(key, val):
    
    l = ['any', 'all', 'every','the','a']
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

def update(pred_pol, stores):
    
    srls = {}
    
    
    new_pol = []
    for rule in pred_pol:
        for k,v in rule.items():
            if k=='action' or k=='decision':
                continue
            store = stores[k]
            if v!='none':
                from_db = store.similarity_search(v,k=1)[0].page_content
            else:
                from_db = v
            rule[k] = postprocess(k, from_db.strip())
        new_pol.append(rule)
        
    for pp in new_pol:
        
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
        
        
    return new_pol, srls

def make_json(labels):
    global count, a, failed_parse
    policies = []
    formatted_list = []
    srls = {}
    for l in labels:
        formatted = format_labels(l)
        formatted_list.append(formatted)
    
    for f in list(set(formatted_list)):
        a+=1
        try:
            label_json = ast.literal_eval(f)
            pp = {'decision': 'allow', 'subject': 'none', 'action': 'none', 'resource': 'none', 'condition': 'none', 'purpose': 'none'}
            for key,val in label_json.items():
                key = key.strip()
                if key in pp:
                    pp[key] = postprocess(key, val.strip())
            policies.append(pp)
            
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
        
        except:
            count+=1
            failed_parse.append([labels, f])
            continue
        
    p = []
    for pol in policies:
        if pol not in p:
            p.append(pol)
        
    return p, srls


def make_json_sar(labels):
    global count, a, failed_parse
    ents = {'subject': [], 'action': [], 'resource': []}
    srls = {}
    policies = []
    formatted_list = []
    for l in labels:
        formatted = format_labels(l)
        formatted_list.append(formatted)
    

    for f in list(set(formatted_list)):
        a += 1
        try:
            label_json = ast.literal_eval(f)
            pp = {'subject': 'none', 'action': 'none', 'resource': 'none'}
            
            for key, val in label_json.items():
                key = key.strip()
                if key in pp:
                    nval = postprocess(key, val.strip())
                    pp[key] = nval

                    ## Added
                    if nval not in ents[key] and nval!='none':
                        ents[key].append(nval)
            policies.append(pp)

        except:
            count += 1
            # print(labels)
            failed_parse.append([labels, f])
            continue

    p = []
    
    # Sanity check
    for k,v in ents.items():
        ents[k] = list(set(v))
    
    for pol in policies:
        if pol not in p:
            p.append(pol)
            
    for pp in p:
        action = pp['action']
        if action not in srls:
            srls[action] = {'subject': [], 'resource': []}
            if pp['subject']!='none':
                srls[action]['subject'].append(pp['subject'])
            if pp['resource']!='none':
                srls[action]['resource'].append(pp['resource'])
        else:
            if pp['subject']!='none':
                srls[action]['subject'].append(pp['subject'])
            if pp['resource']!='none':
                srls[action]['resource'].append(pp['resource'])
        

    return p, ents, srls

            
def process_label(result, mode='sarcp'):
    res = []
    if (len(result) > 0):
        for p in result:
            ind = p.split(" | ")
            if (len(ind) == 1):
                res.append(ind[0])
            else:
                for i in range(len(ind)):
                    if (i==0 and ind[i][-1]!="}"):
                        res.append(ind[i]+"}")
                    elif (i == len(ind)-1 and ind[i][0]!="{"):
                        res.append("{" + ind[i])
                    else:
                        res.append("{" + ind[i] + "}")
    nres = list(set(res))
    if mode == 'sar':
        return (make_json_sar(nres))
    return(make_json(nres))

def update(pred_pol, stores):
    srls = {}
    unique_pols = []
    
    new_pol = []
    for rule in pred_pol:
        for k,v in rule.items():
            if k=='action' or k=='decision':
                continue
            store = stores[k]
            if v!='none':
                from_db = store.similarity_search(v.replace("-", ""),k=1)[0].page_content
            else:
                from_db = v
            rule[k] = postprocess(k, from_db.strip())
        new_pol.append(rule)
        
    
    for r in new_pol:
        if r not in unique_pols:
            unique_pols.append(r)
            
    for pp in unique_pols:
        
        action = pp['action']
        if action not in srls and 'purpose' in pp and 'condition' in pp: 
            srls[action] = {'subject': [], 'resource': [], 'purpose': [], 'condition': []}
        elif action not in srls:
            srls[action] = {'subject': [], 'resource': []}
            
            
        if pp['subject']!='none':
            srls[action]['subject'].append(pp['subject'])
        if pp['resource']!='none':
            srls[action]['resource'].append(pp['resource'])
        if 'purpose' in pp and pp['purpose']!='none':
            srls[action]['purpose'].append(pp['purpose'])
        if 'condition' in pp and pp['condition']!='none':
            srls[action]['condition'].append(pp['condition'])    
        # else:
        #     if pp['subject']!='none':
        #         srls[action]['subject'].append(pp['subject'])
        #     if pp['resource']!='none':
        #         srls[action]['resource'].append(pp['resource'])
        #     if 'purpose' in pp and pp['purpose']!='none':
        #         srls[action]['purpose'].append(pp['purpose'])
        #     if 'condition' in pp and pp['condition']!='none':
        #         srls[action]['condition'].append(pp['condition'])
        
    
    return unique_pols, srls
        
def prepare_inputs(s,l,tokenizer, device = 'cuda:4'):
    
    sent = tokenizer.encode(s, add_special_tokens=False)
    pred = tokenizer.encode(l, add_special_tokens=False)
    pair_token_ids = torch.tensor([[tokenizer.cls_token_id] + sent + [tokenizer.sep_token_id] + pred + [tokenizer.sep_token_id]])
    segment_ids = torch.tensor([[0] * (len(sent) + 2) + [1] * (len(pred) + 1)])
    attention_mask_ids = torch.tensor([[1] * (len(sent) + len(pred) + 3)])
    
    return {
        'input_ids': pair_token_ids.to(device),
        'attention_mask': attention_mask_ids.to(device),
        'token_type_ids': segment_ids.to(device)
    }
    
    
def prepare_inputs_bart(s,l,tokenizer, device = 'cuda:0'):
    
    tokens = tokenizer(s,l,return_tensors='pt')
    
    return {k:v.to(device) for k,v in tokens.items()}


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


def config_loader(config_file):
    with open(config_file, "r") as yamlfile:
        data = yaml.load(yamlfile, Loader=yaml.FullLoader)
        
    print('\n============================================ CONFIGS ============================================\n')
    pprint(data)
    print('\n=================================================================================================\n')
    
    return data


def load_vectorstores(store_location = "../../data/vectorstores"):

    embeddings = HuggingFaceEmbeddings(
        model_name="mixedbread-ai/mxbai-embed-large-v1", model_kwargs={"device": "cuda"}
    )
    VECTORSTORES = {}

    for dset in ["collected", "ibm", "cyber", "acre", "t2p", "misc"]:

        vector_store_subjects = FAISS.load_local(
            f"{store_location}/{dset}/subjects_index",
            embeddings,
            allow_dangerous_deserialization=True,
        )
        vector_store_resources = FAISS.load_local(
            f"{store_location}/{dset}/resources_index",
            embeddings,
            allow_dangerous_deserialization=True,
        )
        vector_store_purposes = FAISS.load_local(
            f"{store_location}/{dset}/purposes_index",
            embeddings,
            allow_dangerous_deserialization=True,
        )
        vector_store_conditions = FAISS.load_local(
            f"{store_location}/{dset}/conditions_index",
            embeddings,
            allow_dangerous_deserialization=True,
        )
        
        vector_store_actions = FAISS.load_local(
            f"{store_location}/{dset}/actions_index",
            embeddings,
            allow_dangerous_deserialization=True,
        )

        stores = {
            "subject": vector_store_subjects,
            "action": vector_store_actions,
            "resource": vector_store_resources,
            "purpose": vector_store_purposes,
            "condition": vector_store_conditions,
        }

        VECTORSTORES[dset] = stores

    return VECTORSTORES

def get_candidates(store, sentence,k=5):
    l = []
    docs = store.similarity_search(sentence,k=k)
    for d in docs:
        l.append(d.page_content)
        
    return l

def get_available_entities(query, store_name, vectorstores, n=5):
    
    entities = {'subject': [], 'action': [], 'resource': [], 'purpose': [], 'condition': []}
    if store_name.endswith('_train'):
        store_name = 'misc'
    
    stores = vectorstores[store_name]
    for key in entities:
        entities[key] = get_candidates(stores[key], query, k=n)

    return entities

def get_srls(policy, post_process = False):
    
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

        
    
    
