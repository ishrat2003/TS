'''
https://github.com/google-research/google-research/tree/master/rouge
'''
import rouge_scorer

class Rouge:

    def __init__(self, model, params):
        self.model = model
        self.params = params
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rouge3', 'rougeL'], use_stemmer=True)
        return
    
    def getScore(self, target, generated):
        score = scorer.score(target, generated)
        return score
