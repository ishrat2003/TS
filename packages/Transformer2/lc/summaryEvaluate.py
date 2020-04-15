from .basicEvaluate import BasicEvaluate
from .peripheral import Peripheral

class SummaryEvaluate(BasicEvaluate):

    def getFileName(self, prefix = ''):
        return self.params.dataset_name + '_' + prefix + '_summary_pos' + str(self.positionContributingFactor) + '_occ' + str(self.occuranceContributingFactor) + '.csv';

    def processItem(self, batch, sourceText, targetText):
        seperator = ' '
        if self.params.display_details:
            print('Batch:::::::::::::::::: ', batch)
            print('Content::: ', sourceText)
            print('Summary::: ', targetText)
        
        row = {}
        # row['main_text'] = sourceText
        # row['summary_text'] = targetText
        
        for posType in self.posGroups:
            self.setAllowedTypes(self.posGroups[posType])
            for topScorePercentage in self.topScorePrecentages:
            
                generatedContributor = self.getContributorByDatasetType(sourceText, topScorePercentage)
                expectedContributor = self.getContributor(targetText, 0, True)
                
                generatedContributor = seperator.join(generatedContributor)
                expectedContributor = seperator.join(expectedContributor)
                
                values = self.evaluate(generatedContributor, expectedContributor, posType, topScorePercentage)
                row.update(values)
                   
        return row
    

    def getContributorByDatasetType(self, text, topScorePercentage = 0.2, allWords = False):
        if(self.params.dataset_name in ['multi_news', 'bhot']):
            contributors = []
            textBlocks = text.split('|||||')
            for textBlock in textBlocks:
                contributors += self.getContributor(textBlock, topScorePercentage, allWords)
            return contributors
        
        return self.getContributor(text, topScorePercentage, allWords)
