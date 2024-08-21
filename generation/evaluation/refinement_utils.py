import torch
import re
from utils import convert_to_sent, prepare_inputs_bart


def verify(nlacp, pred_pol, ver_model, ver_tokenizer):
    pp, _ = convert_to_sent(str(pred_pol))
        
    ver_inp = prepare_inputs_bart(nlacp, pp, ver_tokenizer, "cuda:0")
    pred = ver_model(**ver_inp).logits

    probs = torch.softmax(pred, dim=1)
    pred_class = probs.argmax(axis=-1).item()
    
    return pred_class

def get_refined_policy(instruction, pipe, tokenizer):
    terminators_errors = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    error_generation_args = {
        "max_new_tokens": 1024,
        "return_full_text": False,
        "do_sample": False,
        "temperature": 0.7,
        "eos_token_id": terminators_errors,
        # "stopping_criteria": stopping_criteria
    }
    
    error_messages = [
        {'role': 'system', 'content': ''},
        {'role': 'user', 'content': instruction}
    ]

    response = pipe(error_messages, **error_generation_args)
    
    if ("### Corrected" in response[0]['generated_text'].strip()):

        resp = response[0]['generated_text'].strip().split('### Corrected')[1]
    else:

        resp = response[0]['generated_text'].strip().split('final refined policy')[1]

    pattern = r'\[.*?\]'
    sanitized = re.findall(pattern, resp)
    
    return sanitized[0]

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