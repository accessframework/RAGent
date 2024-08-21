from tabulate import tabulate
from srl_results_sar import SRLEvalSAR, file_reader


datasets = ['t2p', 'acre', 'ibm', 'collected', 'cyber', 'average']

comparison = {j: {k: 0 for k in datasets} for j in ['senna', 'xia', 'ours']}
    
for d in datasets:
    if d == 'average':
        continue
    truth_file = f'results/sar/{d}/{d}_true_srl.json'
    for method in ['senna', 'xia']:
        preds_file = f'results/related_works/{method}/{d}_{method}_pred_srl.json'
        true, preds = file_reader(truth_file, preds_file)
        evaluator = SRLEvalSAR(true,preds)
        p,r,f1 = evaluator.get_f1()
        f = round(f1*100,2)
        comparison[method][d] = f
        comparison[method]['average']+=f
        
    our_preds_file = f'results/sar/{d}/{d}_pred_srl.json'
    true, preds = file_reader(truth_file, our_preds_file)
        
    evaluator1 = SRLEvalSAR(true,preds)
    p,r,f1 = evaluator1.get_f1()
    f = round(f1*100,2)
    comparison['ours'][d] = f
    comparison['ours']['average']+=f
    

table = []

for k,v in comparison.items():
    row = [k]
    for subfield,f1 in v.items():
        if subfield == 'average':
            row.append(round(f1/5.0,2))
        else:
            row.append(f1)
        
    table.append(row)

print('\n================== Result comparison with state-of-the-art (SAR) ===================\n')
print(tabulate(table, headers=datasets, tablefmt="grid"))
print('\n====================================================================================\n')