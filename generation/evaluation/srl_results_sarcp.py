import json
import click 



class SRLEvalSARCP():
    
    def __init__(self, true, preds):
        
        self.fp = 0
        self.fn = 0
        
        self.true, self.preds = true, preds

        self.entities_srls = {'correct': 0, 'possible': 0, 'actual': 0}
        self.component_results = {k: {'tp': 0, 'fp': 0, 'fn': 0, 'p': 0, 'r':0, 'f1': 0} for k in ['subject', 'action', 'resource', 'purpose', 'condition']}
            
    def longest_common_substring(self, str1, str2):
        dp = [[0] * (len(str2) + 1) for _ in range(len(str1) + 1)]
            
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

    def do_overlap(self,x, y, thresh = 0.2):
        
        overlap, start, end = self.longest_common_substring(x, y)
        max_len =  len(x)
        seg_len = end - start +1
        
        if (seg_len/max_len >= thresh):
            return True
            
        return False


    def match_arguments(self, pred_arg_arr, true_arg_arr, arg, thresh = 0.3):
            
        no_overlap = ['subject', 'resource']
            
        if arg in no_overlap:
            for i in pred_arg_arr:
                if i in true_arg_arr:
                    self.entities_srls['correct']+=1
                    self.component_results[arg]['tp']+=1
                    true_arg_arr.remove(i)
                else:
                    self.fp+=1
                    self.component_results[arg]['fp']+=1
        else:
            for i in pred_arg_arr:
                for j in true_arg_arr:
                    if self.do_overlap(j,i,thresh):
                        self.entities_srls['correct']+=1
                        self.component_results[arg]['tp']+=1
                        true_arg_arr.remove(j)
                        break
                    
                else:
                    self.fp+=1
                    self.component_results[arg]['fp']+=1
                    
        if len(true_arg_arr)>0:
            self.fn+=len(true_arg_arr)
            self.component_results[arg]['fn']+=len(true_arg_arr)

    def get_f1(self, calc_comp_f1 = True):
            
        for ts, ps in zip(self.true, self.preds):
                    
            for act,v in ps.items():
                if act in ts:
                    self.component_results['action']['tp']+=1
                    tsv = ts[act]
                    for args, vs in v.items():
                        ltsv = len(tsv[args])
                        self.match_arguments(vs, tsv[args], args, 0.5)
                        self.entities_srls['possible']+=ltsv
                        self.entities_srls['actual']+=len(vs)
                                
                else: ## Over generated (False positive predicates)
                    self.component_results['action']['fp']+=1
                    for ks, vs in v.items():
                        self.entities_srls['actual']+=len(vs)
                        self.component_results[ks]['fp']+=len(vs)
                        self.fp+=len(vs)
                        
                    
            for k,v in ts.items():
                if k not in ps: ## Under generated (False negative predicates)
                    self.component_results['action']['fn']+=1
                    for ks, vs in v.items():
                        self.entities_srls['possible']+=len(vs)
                        self.component_results[ks]['fn']+=len(vs)
                        self.fn+=len(vs)
                                
        p = self.entities_srls['correct']/self.entities_srls['actual']
        r = self.entities_srls['correct']/self.entities_srls['possible']

        tp = self.entities_srls['correct']
        p1 = tp/(tp+self.fp)
        r1 = tp/(tp+self.fn)

        f1 = 2*p1*r1/(p1+r1) if (p1 + r1) > 0 else 0
        
        if calc_comp_f1:
            for component in self.component_results:
                tp = self.component_results[component]['tp']
                fp = self.component_results[component]['fp']
                fn = self.component_results[component]['fn']
                
                prec = tp/(tp+fp) if tp+fp else 0
                rec = tp/(tp+fn) if tp+fn else 0
                
                self.component_results[component]['p'] = prec
                self.component_results[component]['r'] = rec
                
                self.component_results[component]['f1'] = 2*prec*rec/(prec + rec) if prec + rec else 0
        else:
            self.component_results = None
        
        return p1, r1, f1, self.component_results
            
def file_reader(truth_file, preds_file):
    with open(preds_file, "r") as file:
        preds = json.load(file)


    with open(truth_file, "r") as file:
        true = json.load(file)
        
    return true, preds


@click.command()
@click.option('--gt', default='demo_sar_true_srl.json', 
              help='Ground truth file',
              show_default=True,
              required=True,type=click.Choice(['t2p', 'acre', 'ibm', 'collected', 'cyber','overall'], case_sensitive=False))
@click.option('--pred', default='demo_sar_pred_srl.json', 
              help='Predictions file',
              show_default=True,
              )
def main(gt, pred, calculate_component_f1 = True):
    true, preds = file_reader(gt, pred)
    evaluator = SRLEvalSARCP(true, preds)
    p1, r1, f1, component_results = evaluator.get_f1(calculate_component_f1)
    
    print("\n==================== Results (SARCP) ====================\n")
    
    
    print(f'Ground truth file: {gt}\nPredictions file: {pred}\n')
    print(f'Precision: {p1}\nRecall: {r1}\nF1: {f1}\n')
    if calculate_component_f1:
        print('-------------------- Component results --------------------\n')
        print(component_results)
        print('\n-----------------------------------------------------------\n')
    print("===========================================================\n")

if __name__=='__main__':
    
    main()


    
    
