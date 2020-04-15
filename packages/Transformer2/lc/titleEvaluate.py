from .basicEvaluate import BasicEvaluate
from .peripheral import Peripheral

class TitleEvaluate(BasicEvaluate):

    def getFileName(self, prefix = ''):
        return self.params.dataset_name + '_' + prefix + '_title_pos' + str(self.positionContributingFactor) + '_occ' + str(self.occuranceContributingFactor) + '.csv';

    def setAllowedTypes(self, allowedTypes):
        self.allowedTypes = allowedTypes
        return
    
    def process(self):
        self.initInfo()
        
        data = self.dataset.get()
        
        for item in data:
            row = self.processItemForTopic(0, item)
            self.file.write(row)
            self.info['total'] += 1
            print(self.info['total'])
            
        self.summarizeInfo()
        return
    
    def processItemForTopic(self, batch, item):
        source, label = item
        label = label.numpy().decode("utf-8")
        sourceRaw = source.numpy()
        sourceText = self.dataset.getText(sourceRaw)

        seperator = ' '
        if self.params.display_details:
            print('Batch:::::::::::::::::: ', batch)
            print('Content::: ', sourceText)
        
        row = {}
        
        for posType in self.posGroups:
            self.setAllowedTypes(self.posGroups[posType])
            expectedContributor = self.dataset.getTitle(source.numpy())
            row['title'] = expectedContributor
            if self.params.display_details:
                print('Title topics::: ', expectedContributor)
                
            for topScorePercentage in self.topScorePrecentages:
                cwrGeneratedContributor = self.getContributor(sourceRaw.decode("utf-8"), topScorePercentage)
                cwrGeneratedContributor = seperator.join(cwrGeneratedContributor)
                values = self.evaluate(cwrGeneratedContributor, expectedContributor, posType, topScorePercentage)
                row.update(values)
                
        return row
    