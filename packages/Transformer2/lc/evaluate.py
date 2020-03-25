from evaluation.rouge import Rouge
from .peripheral import Peripheral
import os
from utility.file import File
from datetime import datetime

class Evaluate:

    def __init__(self, dataset, params):
        self.dataset = dataset
        self.params = params
        self.allowedTypes = ['NN', 'NNP', 'NNS', 'NNPS']
        self.rouge = Rouge( self.params, ['rouge1'])
        self.file = self.getFile()
        self.posGroups = {}
        self.posGroups['n'] = ['NN', 'NNP', 'NNS', 'NNPS']
        self.posGroups['adj'] = ['JJ', 'JJR', 'JJS']
        self.posGroups['nAdj'] = ['NN', 'NNP', 'NNS', 'NNPS', 
                                  'JJ', 'JJR', 'JJS']
        self.posGroups['v'] = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
        self.posGroups['adv'] = ['RB', 'RBR', 'RBS']
        self.posGroups['vAdv'] = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
                                  'RB', 'RBR', 'RBS']
        self.posGroups['nAdjAdvV'] = ['NN', 'NNP', 'NNS', 'NNPS', 
                                      'JJ', 'JJR', 'JJS',
                                      'RB', 'RBR', 'RBS', 
                                      'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
        self.posGroups['all'] = ["CC","CD","DT","EX","FW","IN","JJ","JJR","JJS","LS","MD",
                                 "NN","NNS","NNP","NNPS","PDT","POS","PRP","PRP$","RB","RBR",
                                 "RBS","RP","SYM","TO","UH","VB","VBD","VBG","VBN","VBP","VBZ",
                                 "WDT","WP","WP$","WRB"]
        self.topScorePrecentages = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        return
    
    def getFile(self):
        now = datetime.now()
        dateString = now.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.params.data_directory, 'cwr', self.params.dataset_name + str(dateString) + '.csv')
        file = File(path)
        return file

    def setAllowedTypes(self, allowedTypes):
        self.allowedTypes = allowedTypes
        return

    def process(self, store = True):

        self.initInfo()
                    
        for (batch, (source, target)) in enumerate(self.dataset):
            sourceText = source.numpy().decode('utf-8')
            targetText = target.numpy().decode('utf-8')
            row = self.processItem(batch, sourceText, targetText)
            print('===================================================')
            print(row)
            self.file.write(row)
            self.info['total'] += 1
            
            
        self.summarizeInfo()
        return
    
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
            
                generatedContributor = self.getContributor(sourceText, topScorePercentage)
                expectedContributor = self.getContributor(targetText, 0, True)
                
                generatedContributor = seperator.join(generatedContributor)
                expectedContributor = seperator.join(expectedContributor)
                
                evaluationScore = self.rouge.getScore(expectedContributor, generatedContributor)
                
                suffix = self.getSuffix(posType, topScorePercentage)
                row['expected_summary_' + suffix] = expectedContributor
                row['generated_contributor_' + suffix] = generatedContributor
                row['rouge1_precision_' + suffix] = evaluationScore['rouge1']['precision']
                row['rouge1_recall_' + suffix] = evaluationScore['rouge1']['recall']
                row['rouge1_fmeasure_' + suffix] = evaluationScore['rouge1']['fmeasure']
                
                self.info['total_precision_' + suffix] += evaluationScore['rouge1']['precision']
                self.info['total_recall_' + suffix] += evaluationScore['rouge1']['recall']
                self.info['total_fmeasure_' + suffix] += evaluationScore['rouge1']['fmeasure']
                
        
        return row
    
    def initInfo(self):
        self.info = {}
        self.info['total'] = 1
        
        for posType in self.posGroups:
            for topScorePercentage in self.topScorePrecentages:
                suffix = self.getSuffix(posType, topScorePercentage)
                self.info['total_precision_' + suffix] = 0
                self.info['total_recall_' + suffix] = 0
                self.info['total_fmeasure_' + suffix] = 0
                
        return
    
    def summarizeInfo(self):
        for posType in self.posGroups:
            for topScorePercentage in self.topScorePrecentages:
                suffix = self.getSuffix(posType, topScorePercentage)
                self.info['avg_precision_' + suffix] = self.info['total_precision_' + suffix] / self.info['total']
                self.info['avg_recall_' + suffix] = self.info['total_recall_' + suffix] / self.info['total']
                self.info['avg_fmeasure_' + suffix] = info['total_fmeasure_' + suffix] / info['total']
                
                del self.info['total_precision_' + suffix]
                del self.info['total_recall_' + suffix]
                del self.info['total_fmeasure_' + suffix]
                
        print(self.info)
        return
    
    def getSuffix(self, posType, topScorePercentage):
        return posType + '_' + str(topScorePercentage)

    def getContributorByDatasetType(self, text, topScorePercentage = 0.2, allWords = False):
        if(self.params.dataset_name in ['multi_news', 'bhot']):
            contributors = []
            textBlocks = text.split('|||||')
            for textBlock in textBlocks:
                contributors += self.getContributor(textBlock, topScorePercentage, allWords)
            return contributors
        
        return self.getContributor(text, topScorePercentage, allWords)
    
    def getContributor(self, text, topScorePercentage = 0.2, allWords = False):
        minAllowedScore = 0 if allWords else 0.1

        peripheralProcessor = Peripheral(text)
        peripheralProcessor.setAllowedPosTypes(self.allowedTypes)
        peripheralProcessor.setPositionContributingFactor(1)
        peripheralProcessor.setOccuranceContributingFactor(1)
        peripheralProcessor.setProperNounContributingFactor(1)
        peripheralProcessor.setTopScorePercentage(topScorePercentage)
        peripheralProcessor.setFilterWords(minAllowedScore)
        peripheralProcessor.loadSentences(text)
        peripheralProcessor.loadFilteredWords()
        peripheralProcessor.train()
        if allWords:
            featuredWords = peripheralProcessor.getFilteredWords()
            return list(featuredWords.keys())

        peripheralProcessor.getPoints()
        return peripheralProcessor.getContrinutors()
