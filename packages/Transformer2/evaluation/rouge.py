'''
https://github.com/google-research/google-research/tree/master/rouge
'''
import rouge_scorer


class Rouge:

    def __init__(self, model):
        self.model = model
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        return
    
    def getScore(self, target, generated):
        score = scorer.score(target, generated)
        return score
    
    def evaluate(self, dataset):
        for (batch, (source, target)) in enumerate(dataset):
            
            if (self.params.display_details == True) :
                print('Batch: ', batch)
                print('source', source.shape)
                print('target', target.shape)
            trainStep(source, target)
            self.endBatch(batch, epoch)

            self.endEpoch(batch, epoch)
        return