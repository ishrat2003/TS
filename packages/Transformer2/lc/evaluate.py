from evaluation.rouge import Rouge
from .peripheral import Peripheral

class Evaluate:

    def __init__(self, dataset, params):
        self.dataset = dataset
        self.params = params
        self.allowedTypes = ['NN', 'NNP', 'NNS', 'NNPS']
        self.rouge = Rouge( self.params)
        return

    def setAllowedTypes(self, allowedTypes):
        self.allowedTypes = allowedTypes
        return

    def process(self):
        for (batch, (source, target)) in enumerate(self.dataset):
            print('Batch:::::::::::::::::: ', batch)
            source = source.numpy().decode('utf-8')
            print('Content::: ', source)
            sourceContributor = self.getContributor(source)
            print('Contributor::: ', sourceContributor)
                    
            target = target.numpy().decode('utf-8')
            print('Summary::: ', target)
            summaryContributor = self.getContributor(target)
            print('Contributor::: ', summaryContributor)

            seperator = ' '
            print('Rouge::', self.rouge.getScore(seperator.join(summaryContributor), seperator.join(sourceContributor)))
        return

    def getContributor(self, text):
        peripheralProcessor = Peripheral(text)
        peripheralProcessor.setAllowedPosTypes(self.allowedTypes)
        peripheralProcessor.setPositionContributingFactor(1)
        peripheralProcessor.setOccuranceContributingFactor(1)
        peripheralProcessor.setProperNounContributingFactor(1)
        peripheralProcessor.setTopScorePercentage(0.6)
        peripheralProcessor.setFilterWords(0.1)
        peripheralProcessor.loadSentences(text)
        peripheralProcessor.loadFilteredWords()
        peripheralProcessor.train()
        peripheralProcessor.getPoints()
        return peripheralProcessor.getContrinutors()
