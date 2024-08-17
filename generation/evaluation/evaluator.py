## This script is inspired by: https://github.com/davidsbatista/NER-Evaluation


from collections import namedtuple
from copy import deepcopy
import logging
import json

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level="INFO",
)
from srl_results_sarcp import SRLEvalSARCP

class AccessEvaluator():

    def __init__(self, true, pred, truesrls = [], predsrls = [], inputs=None, confs = None, true_confs=None, cats = None):
        """
        
        true: [[{'decision': 'allow', 'subject': 'a', ...}, {'acp': 'yes', 'decision': 'allow', 'subject': 'a', ...}], [{...}], ]
        pred: true: [[{'decision': 'allow', 'subject': 'a', ...}, {'acp': 'yes', 'decision': 'allow', 'subject': 'a', ...}], [{...}], ]
        
        """
        
        self.test_c = 0
        self.partials = []

        if len(true) != len(pred):
            raise ValueError("Number of predicted documents does not equal true")
        
        if inputs == None:
            self.inputs = [None]*len(true)
        else:
            self.inputs = inputs
            
            
        if confs == None:
            self.confs = [None]*len(true)
        else:
            self.confs = confs
            
        if true_confs == None:
            self.true_confs = [None]*len(true)
        else:
            self.true_confs = true_confs
            
        if cats == None:
            self.cats = [None]*len(true)
        else:
            self.cats = cats
            

        self.true = true
        self.pred = pred
        self.incorrect_preds = []

        self.truesrls = truesrls
        self.predsrls = predsrls
        
        self.entities_srls = {'correct': 0, 'possible': 0, 'actual': 0}
        self.purcount, self.condcount = 0,0
        self.purcorrectcount, self.condcorrectcount = 0,0
        
        # Setup dict into which metrics will be stored.

        self.metrics_results = {
            "correct": 0,
            "incorrect": 0,
            "missed": 0,
            "spurious": 0,
            "possible": 0,
            "actual": 0,
            "precision": 0,
            "recall": 0,
            "f1-score": 0
        }

        # Copy results dict to cover the four schemes.
        
        self.schemas = ['strict', 'ent_type']

        self.results = {
            "strict": deepcopy(self.metrics_results),
            "ent_type": deepcopy(self.metrics_results)
            }
        
        
    def evaluate(self):
        logging.info("Evaluating ...")
        
        # Iterate over the list of lists. One list represent a one NLACP containing multiple policies (multiple dictionaries in one list)
        for true_acp, pred_acp, sent, conf, tc, cat in zip(self.true, self.pred, self.inputs, self.confs, self.true_confs, self.cats):
            
            tmp_results = self.compute_metrics(true_acp, pred_acp, sent, conf, tc, cat)
            
            for eval_schema in self.schemas:
                for metric in self.results[eval_schema]:
                    self.results[eval_schema][metric] += tmp_results[eval_schema][metric]

                # Calculate global precision and recall

            self.results = self.compute_precision_recall_wrapper(self.results)
            
        evaluator = SRLEvalSARCP(self.truesrls, self.predsrls)
        self.results['srl'] = {"precision": 0, "recall": 0, "f1": 0, "components": None}
        self.results['srl']['precision'], self.results['srl']['recall'], self.results['srl']['f1'], self.results['srl']['components'] = evaluator.get_f1(True)
            
            
        return self.results, self.incorrect_preds
        
        
            
            
    def longest_common_substring(self, str1, str2):
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
    
    def do_overlap(self, x, y, thresh = 0.5):
        
        
        overlap, start, end = self.longest_common_substring(x, y)
        max_len =  len(x)
        seg_len = end - start +1
        
        if (seg_len/max_len >= thresh):
            return True
        
        return False
    

    def compute_metrics(self, true_acp, pred_acp, sent, conf, tc, cat):
        
        
        l = [sent, true_acp, pred_acp, conf, tc, cat]

        '''
        true_acp: [{'acp': 'yes', 'decision': 'allow', 'subject': 'a', ...}, {'acp': 'yes', 'decision': 'allow', 'subject': 'a', ...}]
        pred_acp: [{'acp': 'yes', 'decision': 'allow', 'subject': 'a', ...}, {'acp': 'yes', 'decision': 'allow', 'subject': 'a', ...}]
        
        '''

        eval_metrics = {"correct": 0, "incorrect": 0, "missed": 0, "spurious": 0, "precision": 0, "recall": 0, "f1-score": 0}

        # overall results
        
        evaluation = {
            "strict": deepcopy(eval_metrics),
            "ent_type": deepcopy(eval_metrics)
        }

        # keep track of entities that overlapped

        true_which_overlapped_with_pred = []
        exact_match = []
        
        for pred in pred_acp:
            
            d_pred = pred['decision']
            s_pred = pred['subject']
            a_pred = pred['action']
            r_pred = pred['resource']
            p_pred = pred['purpose']
            c_pred = pred['condition']
            
            if pred in true_acp:
                exact_match.append(pred)
        
        
        for pred in pred_acp:
            
            d_pred = pred['decision']
            s_pred = pred['subject']
            a_pred = pred['action']
            r_pred = pred['resource']
            p_pred = pred['purpose']
            c_pred = pred['condition']
            
            found = False
            
            if pred in true_acp:
                true_which_overlapped_with_pred.append(pred)
                evaluation["strict"]["correct"] += 1
                evaluation["ent_type"]["correct"] += 1
                
                found = True
                
            else:
                for true in true_acp:
                    
                    # acp_true = true['acp']
                    d_true = true['decision']
                    s_true = true['subject']
                    a_true = true['action']
                    r_true = true['resource']
                    p_true = true['purpose']
                    c_true = true['condition']
                    
                        
                    if (true not in exact_match) and (true not in true_which_overlapped_with_pred) and (self.do_overlap(d_true, d_pred,1) and self.do_overlap(s_true, s_pred,1) and self.do_overlap(a_true, a_pred,1) and
                    self.do_overlap(r_true, r_pred, 1) and self.do_overlap(p_true, p_pred, 0.2) and
                    self.do_overlap(c_true, c_pred, 0.2)):
                            
                        true_which_overlapped_with_pred.append(true)
                            
                        evaluation["strict"]["incorrect"] += 1
                        evaluation["ent_type"]["correct"] += 1
                        
                        self.partials.append([sent, true, pred])
                            
                        found = True
                        break
                    
                if not found:
                    
                    # Over generated
                    
                    evaluation["strict"]["spurious"] += 1
                    evaluation["ent_type"]["spurious"] += 1
                    
                    if l not in self.incorrect_preds:
                        self.incorrect_preds.append(l)
                        
                    
                    
        for true in true_acp:
            
            if true in true_which_overlapped_with_pred:
                continue
            else:
                # Under generated
                evaluation["strict"]["missed"] += 1
                evaluation["ent_type"]["missed"] += 1
                
                if l not in self.incorrect_preds:
                    self.incorrect_preds.append(l)
                
                
        for eval_type in evaluation:
            evaluation[eval_type] = self.compute_actual_possible(evaluation[eval_type])
                    

        
        return evaluation
    
    
    def compute_actual_possible(self, results):
        
        """
        Takes a result dict that has been output by compute metrics.
        Returns the results dict with actual, possible populated.

        When the results dicts is from partial or ent_type metrics, then
        partial_or_type=True to ensure the right calculation is used for
        calculating precision and recall.
        """
        correct = results["correct"]
        incorrect = results["incorrect"]
        missed = results["missed"]
        spurious = results["spurious"]

        possible = correct + incorrect + missed

        actual = correct + incorrect + spurious

        results["actual"] = actual
        results["possible"] = possible
        
        return results
    
    
    def compute_precision_recall(self, results):
        """
        Takes a result dict that has been output by compute metrics.
        Returns the results dict with precison and recall populated.

        When the results dicts is from partial or ent_type metrics, then
        partial_or_type=True to ensure the right calculation is used for
        calculating precision and recall.
        """

        actual = results["actual"]
        possible = results["possible"]
        correct = results["correct"]

        precision = correct / actual if actual > 0 else 0
        recall = correct / possible if possible > 0 else 0

        results["precision"] = precision
        results["recall"] = recall
        results["f1-score"] = (2*precision*recall)/float(precision + recall) if (precision + recall) > 0 else 0

        return results
    
    def compute_precision_recall_wrapper(self, results):
        """
        Wraps the compute_precision_recall function and runs on a dict of results
        """

        result = {key: self.compute_precision_recall(value) for key, value in results.items() if key!='srl'}

        results = {**result}

        return results
                    
                