from evaluation.rouge import Rouge
import os
from utility.file import File
from datetime import datetime
from topic.lda import LDA 

class Evaluate:

    def __init__(self, dataset, params):
        self.dataset = dataset
        self.params = params
        self.allowedTypes = ['NN', 'NNP', 'NNS', 'NNPS']
        self.rouge = Rouge( self.params, ['rouge1'])
        self.file = self.getFile()
        self.lda = LDA(self.dataset, os.path.join(self.params.data_directory, self.params.dataset_name))
        return
    
    def getFile(self):
        now = datetime.now()
        dateString = now.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.params.data_directory, self.params.dataset_name + '__lda_' + str(dateString) + '.csv')
        file = File(path)
        return file


    def process(self, store = True):
        self.initInfo()
        data = self.dataset.get()
        for item in data:
            row = self.processItem(0, item)
            self.file.write(row)
            self.info['total'] += 1
            print(self.info['total'])
            
        self.summarizeInfo()
        return
    
    def processItem(self, batch, item):
        source, label = item
        label = label.numpy().decode("utf-8")
        sourceRaw = source.numpy()
        sourceText = self.dataset.getText(sourceRaw)

        seperator = ' '
        if self.params.display_details:
            print('Batch:::::::::::::::::: ', batch)
            print('Content::: ', sourceText)
        
        row = {}

        expectedContributor = self.dataset.getTitle(source.numpy())

        totalExpected = len(expectedContributor.split(' '))
        if totalExpected < 2:
            totalExpected = 5

        ldaGeneratedTopics = self.lda.predictedTopics(sourceText.split(' '), label, limit = totalExpected)
        ldaGeneratedTopics = seperator.join(ldaGeneratedTopics)
   
        row['title'] = expectedContributor
        row['lda'] = ldaGeneratedTopics
        evaluationScore = self.rouge.getScore(expectedContributor, ldaGeneratedTopics)
            
        row['rouge1_precision_lda'] = evaluationScore['rouge1']['precision']
        row['rouge1_recall_lda'] = evaluationScore['rouge1']['recall']
        row['rouge1_fmeasure_lda'] = evaluationScore['rouge1']['fmeasure']
            
        self.info['lda_total_precision'] += evaluationScore['rouge1']['precision']
        self.info['lda_total_recall'] += evaluationScore['rouge1']['recall']
        self.info['lda_total_fmeasure'] += evaluationScore['rouge1']['fmeasure']
            
        if self.params.display_details:
            print('Title topics::: ', expectedContributor)
            print('LDA topics::: ', ldaGeneratedTopics)
                   
        return row
    
 
    def initInfo(self):
        self.info = {}
        self.info['lda_total_precision'] = 0
        self.info['lda_total_recall'] = 0
        self.info['lda_total_fmeasure'] = 0
        
        self.info['lda_avg_precision'] = 0
        self.info['lda_avg_recall'] = 0
        self.info['lda_avg_fmeasure'] = 0
        
        self.info['total'] = 1
        return
    
    def summarizeInfo(self):
        self.info['lda_avg_precision'] = self.info['lda_total_precision'] / self.info['total']
        self.info['lda_avg_recall'] = self.info['lda_total_recall'] / self.info['total']
        self.info['lda_avg_fmeasure'] = self.info['lda_total_fmeasure'] / self.info['total']
        
        del self.info['lda_total_precision']
        del self.info['lda_total_recall']
        del self.info['lda_total_fmeasure']
        return
