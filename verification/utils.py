import ast
from evaluate import load
import random
import torch


basic_template = "{subject} is {decision} {action} {resource}"
basic_template_subject = "{subject} is {decision} {action}"
basic_template_resource = "{resource} is {decision} be {action}"

purpose_template = "{template} for the purpose of {purpose}"
condition_template = "{template} if {condition}"
sarcp_template = "{template} for the purpose of {purpose}, if {condition}"

allow_sents = ['allowed to', 'able to', 'shall', 'authorized to']
deny_sents = ['prohibited to', 'not able to', 'shall not', 'unauthorized to']

def convert_to_sent(sample):
    error = 0
    # sample = ast.literal_eval(str(sample))
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

def preprocess_logits_for_metrics_bart(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    # pred_ids = torch.argmax(logits[0], dim=-1)
    predictions = torch.argmax(logits[0], dim=-1)
    return predictions


def compute_metrics(eval_pred):
    load_accuracy = load("accuracy")
    load_f1 = load("f1")
   
    labels = eval_pred.label_ids
    predictions = eval_pred.predictions
   
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1_micro = load_f1.compute(predictions=predictions, references=labels, average = 'micro')["f1"]
    f1_macro = load_f1.compute(predictions=predictions, references=labels, average = 'macro')["f1"]
    f1_weighted = load_f1.compute(predictions=predictions, references=labels, average = 'weighted')["f1"]
    return {"accuracy": accuracy, "f1-micro": f1_micro, "f1-macro": f1_macro, "f1-weighted": f1_weighted}

a,count = 0,0
failed_parse = []

def create_out_string(inp):
    s = "{"
    for e in inp:
        for k,v in e.items():
            s+=f"{k}: {v}; "
            
        s = s[:-2] + " | "
        
    return s[:-3] + "}"

def format_labels(label: str):
    nlabel =  label.lower().replace('"', '').replace(';',',').replace('decision: ', "\'decision\': '").replace('subject: ', "\'subject\': '").replace('action: ', "\'action\': '").replace('resource: ', "\'resource\': '").replace('condition: ', "\'condition\': '").replace('purpose: ', "\'purpose\': '").replace(",", "',").replace('}', "'}").replace("'s ", "\\'s ")
    i = nlabel.find('decision')
    return ("{'" + nlabel[i:])

def make_json(labels):
    global count, a, failed_parse
    policies = []
    formatted_list = []
    for l in labels:
        formatted = format_labels(l)
        formatted_list.append(formatted)
    
    for f in list(set(formatted_list)):
        a+=1
        try:
            label_json = ast.literal_eval(f)
            pp = {'decision': 'allow', 'subject': 'none', 'action': 'none', 'resource': 'none', 'condition': 'none', 'purpose': 'none'}
            for key,val in label_json.items():
                if key in pp:
                    pp[key] = val.strip()
            policies.append(pp)
        
        except:
            count+=1
            failed_parse.append([labels, f])
            continue
        
    p = []
    for pol in policies:
        if pol not in p:
            p.append(pol)
        
    return p
            
            
def process_label(result):
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
    return(make_json(nres))


def longest_common_substring(str1, str2):
        # Initialize a matrix to store the lengths of common substrings
        # dp[i][j] will store the length of the longest common substring ending at str1[i-1] and str2[j-1]
    dp = [[0] * (len(str2) + 1) for _ in range(len(str1) + 1)]
        
        # Variables to store the length of the longest common substring and its ending index
    longest_substring_length = 0
    longest_substring_end_index = 0
        
        # Fill the matrix
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > longest_substring_length:
                    longest_substring_length = dp[i][j]
                    longest_substring_end_index = i - 1
            else:
                dp[i][j] = 0
        
        # Extract the longest common substring
    longest_substring_start_index = longest_substring_end_index - longest_substring_length + 1
    longest_substring = str1[longest_substring_start_index:longest_substring_end_index + 1]
        
    return longest_substring, longest_substring_start_index, longest_substring_end_index

def do_overlap(x, y, thresh = 1):
        
    overlap, start, end = longest_common_substring(x.lower(), y.lower())
    max_len =  len(x) #max(len(x), len(y))
        
    seg_len = end - start +1
        
    if (seg_len/max_len >= thresh):
        return True
        
    return False


def is_equal(preds, labels):
    pcopy = preds.copy()
    lcopy = labels.copy()
    
    found = []
    
    if len(preds) != len(labels):
        return False
    else:
        for pred in preds:
            d = pred['decision']
            s = pred['subject']
            a = pred['action']
            r = pred['resource']
            p = pred['purpose']
            c = pred['condition']
            
            if pred in labels and pred not in found:
                found.append(pred)
                lcopy.remove(pred)
                pcopy.remove(pred)
                
        for pred in preds:
            if pred not in found:
                d = pred['decision']
                s = pred['subject']
                a = pred['action']
                r = pred['resource']
                p = pred['purpose']
                c = pred['condition']
                
                for l in labels:
                    if l in found:
                        continue
                    dl = l['decision']
                    sl = l['subject']
                    al = l['action']
                    rl = l['resource']
                    pl = l['purpose']
                    cl = l['condition']
                    
                    if do_overlap(dl, d) and do_overlap(sl, s) and do_overlap(al, a) and do_overlap(rl, r, 0.8) and do_overlap(pl, p, 0.2) and do_overlap(cl, c, 0.2):
                        found.append(l)
                        lcopy.remove(l)
                        pcopy.remove(pred)
                        
                        break

        if len(pcopy) == len(lcopy) == 0:
            return True
        else:
            return False
        
def prepare_inputs_bart(s,l,tokenizer, device = 'cuda:0'):
    
    tokens = tokenizer(s,l,return_tensors='pt')
    
    return {k:v.to(device) for k,v in tokens.items()}
