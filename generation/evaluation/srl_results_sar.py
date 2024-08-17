import json
import click 

class SRLEvalSAR():
    
    def __init__(self, ground_truths, preds):

        self.fp = 0
        self.fn = 0

        self.entities_srls = {'correct': 0, 'possible': 0, 'actual': 0}
        
        self.true, self.preds = ground_truths, preds

    def match_arguments(self,pred_arg_arr, true_arg_arr):
            
        for i in pred_arg_arr:
            if i in true_arg_arr:
                self.entities_srls['correct']+=1
                true_arg_arr.remove(i)
            else:
                self.fp+=1
                    
        if len(true_arg_arr)>0:
            self.fn+=len(true_arg_arr)

    def get_f1(self):

        for ts, ps in zip(self.true, self.preds):
                    
            for act,v in ps.items():
                if act in ts:
                    tsv = ts[act]
                    for args, vs in v.items():
                        ltsv = len(tsv[args])
                        self.match_arguments(vs, tsv[args])
                        self.entities_srls['possible']+=ltsv
                        self.entities_srls['actual']+=len(vs)
                                
                else: ## Over generated (False positive predicates)
                    for ks, vs in v.items():
                        self.entities_srls['actual']+=len(vs)
                        self.fp+=len(vs)
                        
                    
            for k,v in ts.items():
                if k not in ps: ## Under generated (False negative predicates)
                    for ks, vs in v.items():
                        self.entities_srls['possible']+=len(vs)
                        self.fn+=len(vs)
                                
        p = self.entities_srls['correct']/self.entities_srls['actual']
        r = self.entities_srls['correct']/self.entities_srls['possible']

        tp = self.entities_srls['correct']
        p1 = tp/(tp+self.fp)
        r1 = tp/(tp+self.fn)

        f1 = 2*p1*r1/(p1+r1) if (p1 + r1) > 0 else 0
        
        return p1,r1,f1
        
def file_reader(truth_file, preds_file):
    with open(preds_file, "r") as file:
        preds = json.load(file)


    with open(truth_file, "r") as file:
        true = json.load(file)
        
    return true, preds

@click.command()
@click.option('--mode', default='t2p', 
              help='Mode of training (document-fold you want to evaluate the trained model on)',
              show_default=True,
              required=True,
              type=click.Choice(['t2p', 'acre', 'ibm', 'collected', 'cyber'], case_sensitive=False))
def main(mode):
    
    ground_truth = f'results/sar/true_{mode}_srl.json'
    predictions = f'results/sar/pred_{mode}_srl.json'
    true, preds = file_reader(ground_truth, predictions)
    evaluator = SRLEvalSAR(true, preds)
    p1, r1, f1 = evaluator.get_f1()
    
    print("\n==================== Results (SAR) ========================\n")
    
    
    print(f'Ground truth file: {ground_truth}\nPredictions file: {predictions}\n')
    print(f'Precision: {p1}\nRecall: {r1}\nF1: {f1}\n')
    print("===========================================================\n")

if __name__=='__main__':
    
    main()


    
    
