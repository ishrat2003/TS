from evaluation.rouge import Rouge
from .peripheral import Peripheral
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
        self.fileAvg = self.getFile('avg')
        self.posGroups = {}
        self.posGroups['n'] = ['NN', 'NNP', 'NNS', 'NNPS']
        # self.posGroups['adj'] = ['JJ', 'JJR', 'JJS']
        # self.posGroups['nAdj'] = ['NN', 'NNP', 'NNS', 'NNPS', 
        #                           'JJ', 'JJR', 'JJS']
        # self.posGroups['v'] = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
        # self.posGroups['adv'] = ['RB', 'RBR', 'RBS']
        # self.posGroups['vAdv'] = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
        #                           'RB', 'RBR', 'RBS']
        # self.posGroups['nAdjAdvV'] = ['NN', 'NNP', 'NNS', 'NNPS', 
        #                               'JJ', 'JJR', 'JJS',
        #                               'RB', 'RBR', 'RBS', 
        #                               'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
        # self.posGroups['all'] = ["CC","CD","DT","EX","FW","IN","JJ","JJR","JJS","LS","MD",
        #                          "NN","NNS","NNP","NNPS","PDT","POS","PRP","PRP$","RB","RBR",
        #                          "RBS","RP","SYM","TO","UH","VB","VBD","VBG","VBN","VBP","VBZ",
        #                          "WDT","WP","WP$","WRB"]
        self.topScorePrecentages = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        self.lda = LDA(self.dataset, os.path.join(self.params.data_directory, self.params.dataset_name))
        return
    
    def getFile(self, prefix = ''):
        now = datetime.now()
        dateString = now.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.params.data_directory, 'cwr', self.params.dataset_name + '_' + prefix + '_' + str(dateString) + '_pos10occ0.csv')
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
            # print('===================================================')
            # print(row)
            self.file.write(row)
            self.info['total'] += 1
            
            
        self.summarizeInfo()
        return
    
    def processTopics(self):
        self.initInfo()
        
        data = self.dataset.get()
        
        for item in data:
            row = self.processItemForTopic(0, item)
            # print('===================================================')
            # print(row)
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
            # print(sourceText)
            # print(label)
            totalExpected = len(expectedContributor.split(' '))
            if totalExpected < 2:
                totalExpected = 5
            # print('totalExpected', totalExpected)
            
            ldaGeneratedTopics = self.lda.predictedTopics(sourceText.split(' '), label, limit = totalExpected)
            ldaGeneratedTopics = seperator.join(ldaGeneratedTopics)
            # print('title', expectedContributor)
            # print('ldaGeneratedTopics', ldaGeneratedTopics)
            
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
                
            for topScorePercentage in self.topScorePrecentages:
                cwrGeneratedContributor = self.getContributor(sourceRaw.decode("utf-8"), topScorePercentage)
                cwrGeneratedContributor = seperator.join(cwrGeneratedContributor)
                # print('cwrGeneratedContributor', cwrGeneratedContributor)
                if self.params.display_details:
                    print('CWR topics::: ', cwrGeneratedContributor)
                    
                suffix = self.getSuffix(posType, topScorePercentage)
                row['cwr_' + suffix] = cwrGeneratedContributor

                evaluationScore = self.rouge.getScore(expectedContributor, cwrGeneratedContributor)
                
                row['cwr_rouge1_precision_' + suffix] = evaluationScore['rouge1']['precision']
                row['cwr_rouge1_recall_' + suffix] = evaluationScore['rouge1']['recall']
                row['cwr_rouge1_fmeasure_' + suffix] = evaluationScore['rouge1']['fmeasure']
                
                self.info['total_precision'][suffix] += evaluationScore['rouge1']['precision']
                self.info['total_recall'][suffix] += evaluationScore['rouge1']['recall']
                self.info['total_fmeasure'][suffix] += evaluationScore['rouge1']['fmeasure']
                
        return row
    
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
                
                self.info['total_precision'][suffix] += evaluationScore['rouge1']['precision']
                self.info['total_recall'][suffix] += evaluationScore['rouge1']['recall']
                self.info['total_fmeasure'][suffix] += evaluationScore['rouge1']['fmeasure']
                
        return row
    
    def initInfo(self):
        self.info = {}
        self.info['total_precision'] = {}
        self.info['total_recall'] = {}
        self.info['total_fmeasure'] = {}
        self.info['avg_precision'] = {}
        self.info['avg_recall'] = {}
        self.info['avg_fmeasure'] = {}
        self.info['lda_total_precision'] = 0
        self.info['lda_total_recall'] = 0
        self.info['lda_total_fmeasure'] = 0
        self.info['lda_avg_precision'] = 0
        self.info['lda_avg_recall'] = 0
        self.info['lda_avg_fmeasure'] = 0
        self.info['lda_precision'] = 0
        self.info['lda_recall'] = 0
        self.info['lda_fmeasure'] = 0
        self.info['total'] = 1
        
        for posType in self.posGroups:
            for topScorePercentage in self.topScorePrecentages:
                suffix = self.getSuffix(posType, topScorePercentage)
                self.info['total_precision'][suffix] = 0
                self.info['total_recall'][suffix] = 0
                self.info['total_fmeasure'][suffix] = 0
                
        return
    
    def summarizeInfo(self):
        self.info['lda_avg_precision'] = self.info['lda_total_precision'] / self.info['total']
        self.info['lda_avg_recall'] = self.info['lda_total_recall'] / self.info['total']
        self.info['lda_avg_fmeasure'] = self.info['lda_total_fmeasure'] / self.info['total']
        
        del self.info['lda_total_precision']
        del self.info['lda_total_recall']
        del self.info['lda_total_fmeasure']

        for posType in self.posGroups:
            for topScorePercentage in self.topScorePrecentages:
                suffix = self.getSuffix(posType, topScorePercentage)
                self.info['avg_precision'][suffix] = self.info['total_precision'][suffix] / self.info['total']
                self.info['avg_recall'][suffix] = self.info['total_recall'][suffix] / self.info['total']
                self.info['avg_fmeasure'][suffix] = self.info['total_fmeasure'][suffix] / self.info['total']
                
                del self.info['total_precision'][suffix]
                del self.info['total_recall'][suffix]
                del self.info['total_fmeasure'][suffix]
                
                row = {}
                row['suffix'] = suffix
                row['postype'] = posType
                row['radious'] = 1.0 - topScorePercentage
                row['precision'] = self.info['avg_precision'][suffix]
                row['recall'] = self.info['avg_recall'][suffix]
                row['fmeasure'] = self.info['avg_fmeasure'][suffix]
                row['lda_precision'] = self.info['lda_avg_precision']
                row['lda_recall'] = self.info['lda_avg_recall']
                row['lda_fmeasure'] = self.info['lda_avg_fmeasure']
                self.fileAvg.write(row)
        # print(self.info)
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
        peripheralProcessor.setPositionContributingFactor(10)
        peripheralProcessor.setOccuranceContributingFactor(0)
        peripheralProcessor.setProperNounContributingFactor(0)
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
