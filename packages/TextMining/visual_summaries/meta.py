import lc as LC
from file.writer import Writer
import operator
import utility
from nltk import word_tokenize, pos_tag
import re, math, io, os
import numpy
from nltk.stem.porter import PorterStemmer

class Meta(Writer):
    
    def __init__(self, datasetProcessor, prefix = 'summary'):
        self.prefix = prefix
        super().__init__()
        self.datasetProcessor = datasetProcessor
        self.path = self.datasetProcessor.getPath()
        self.stopWords = utility.Utility.getStopWords()
        self.metaFile = self.getFile(self.prefix + 'meta.csv')
        self.docsSummaryFile = {}
        self.summaryWords = {}
        self.wordKeyByIndex = {}
        self.totalPoints = 0
        self.setAllowedPOSTypes(['NN', 'NNP', 'NNS', 'NNPS'])
        self.punctuationTypes = ['.', '?', '!']
        self.stemmer = PorterStemmer()
        self.minWordSize = 2
        return
    
    
    '''
    1.	CC	Coordinating conjunction
    2.	CD	Cardinal number
    3.	DT	Determiner
    4.	EX	Existential there
    5.	FW	Foreign word
    6.	IN	Preposition or subordinating conjunction
    7.	JJ	Adjective
    8.	JJR	Adjective, comparative
    9.	JJS	Adjective, superlative
    10.	LS	List item marker
    11.	MD	Modal
    12.	NN	Noun, singular or mass
    13.	NNS	Noun, plural
    14.	NNP	Proper noun, singular
    15.	NNPS	Proper noun, plural
    16.	PDT	Predeterminer
    17.	POS	Possessive ending
    18.	PRP	Personal pronoun
    19.	PRP$	Possessive pronoun
    20.	RB	Adverb
    21.	RBR	Adverb, comparative
    22.	RBS	Adverb, superlative
    23.	RP	Particle
    24.	SYM	Symbol
    25.	TO	to
    26.	UH	Interjection
    27.	VB	Verb, base form
    28.	VBD	Verb, past tense
    29.	VBG	Verb, gerund or present participle
    30.	VBN	Verb, past participle
    31.	VBP	Verb, non-3rd person singular present
    32.	VBZ	Verb, 3rd person singular present
    33.	WDT	Wh-determiner
    34.	WP	Wh-pronoun
    35.	WP$	Possessive wh-pronoun
    36.	WRB	Wh-adverb
    '''
    def setAllowedPOSTypes(self, allowedTypes):
        self.allowedPOSTypes = allowedTypes
        return
    
    def process(self):
        self.currentDocIndex = 0
        self.summaryWords = {}
        self.docsSummaryFile = {}
        data = self.datasetProcessor.getTrainingSet()
        for item in data:
            label = self.datasetProcessor.getLabel(item)
            abstractText = self.datasetProcessor.getAbstractText(item)
            #print('Abstract:', abstractText)
            sentences = self.__getSentences(abstractText)
            #print('Sentences:', sentences)
            self.__updateSummaryClusterInfo(sentences, label, abstractText)
            self.currentDocIndex += 1
            #print('currentDocIndex: ', self.currentDocIndex)

        self.__sortSummaryWords()
        for wordKey in self.summaryWords.keys():
            data = self.summaryWords[wordKey]
            #print('data', data)
            del data['doc_references']
            data['groups'] = '_'.join(str(x) for x in data['groups'])
            self.metaFile.write(data)
            
        self.totalPoints = len(self.summaryWords)
        self.__saveSummayVocab()
        self.__saveDocs()
        self.__saveWordCoOccurance()
        
        del self.summaryWords
        del self.docsSummaryFile
        return
    
    def loadSummaryVocab(self):
        self.summaryWords = self._getFromPickel(self.__getVocabPath())
        return self.summaryWords
    
    def loadDocs(self):
        self.docsLCFile = self._getFromPickel(self.__getSummaryDocsPath())
        return self.docsLCFile
    
    def remove(self):
        file = utility.File(self.__getSummaryDocsPath())
        file.remove()
        file = utility.File(self.__getVocabPath())
        file.remove()
        self.metaFile.remove()
        return

    def __getSentences(self, text):
        words = self.__getWords(text)

        self.wordInfo = {}
        sentences = []
        currentSentence = []
        
        for word in words:
            (word, type) = word
            
            word = self.__cleanWord(word)
            if type in self.punctuationTypes:
                sentences.append(currentSentence)
                currentSentence = []
                
            if (len(word) < self.minWordSize) or (word in self.stopWords):
                continue

            wordKey = self.__addWordInfo(word, type)
        
            if wordKey and (wordKey not in currentSentence):
                currentSentence.append(wordKey)

        sentences.append(currentSentence)

        return sentences

    def __addWordInfo(self, word, type):
        if not word or (type not in self.allowedPOSTypes):
            return None

        if word in self.stopWords:
            return None

        localWordInfo = {}
        localWordInfo['pure_word'] = word
        wordKey = self.stemmer.stem(word.lower())
        localWordInfo['stemmed_word'] = wordKey
        localWordInfo['type'] = type
        localWordInfo['number_of_blocks'] = 0
        localWordInfo['groups'] = []
        localWordInfo['doc_references'] = []
        
        if localWordInfo['stemmed_word'] in self.summaryWords.keys():
            self.summaryWords[wordKey]['count'] += 1
            return wordKey
            

        localWordInfo['count'] = 1
        localWordInfo['index'] = len(self.summaryWords)
        self.summaryWords[wordKey] = localWordInfo
        self.wordKeyByIndex[localWordInfo['index']] = wordKey
        return wordKey
    
    def __updateSummaryClusterInfo(self, sentences, lable, summaryText):
        self.docsSummaryFile[lable] = {
            'summary': summaryText,
            'sentences': []
        }
        totalSentences = len(sentences)
        sentenceGroupSpan = math.ceil(totalSentences / 3)
        #print('totalSentences ----', totalSentences)
        #print('sentenceGroupSpan ----', sentenceGroupSpan)

        index = 0
        for sentence in sentences:
            sentenceWithIndecies = []
            for wordKey in sentence:
                group = math.floor(index / sentenceGroupSpan) + 1
                #print('group--', group, 'index --', index)
                if group not in self.summaryWords[wordKey]['groups']:
                    self.summaryWords[wordKey]['groups'].append(group)
                 
                if lable not in self.summaryWords[wordKey]['doc_references']:
                    #print('lable', lable)
                    self.summaryWords[wordKey]['doc_references'].append(lable)
                    self.summaryWords[wordKey]['number_of_blocks'] += 1
                    
                sentenceWithIndecies.append(self.summaryWords[wordKey]['index'])
        
            index += 1        
            if len(sentenceWithIndecies) > 1:
                self.docsSummaryFile[lable]['sentences'].append(sentenceWithIndecies)
        return
    
    def __saveWordCoOccurance(self):
        vectors = numpy.zeros((self.totalPoints, self.totalPoints))
        for label in self.docsSummaryFile.keys():
            sentences = self.docsSummaryFile[label]['sentences']
            
            for sentence in sentences:
                for rowWordIndex in sentence:
                    for columnWordIndex in sentence:
                        vectors[rowWordIndex][columnWordIndex] += 1
                        
        outV = io.open(os.path.join(self.path, self.prefix + '_embedding_vecs.tsv'), 'w', encoding='utf-8')
        outM = io.open(os.path.join(self.path, self.prefix + '_embedding_meta.tsv'), 'w', encoding='utf-8')

        writeHeader = True
        outM.write("word\tcluster\n")

        for index in self.wordKeyByIndex.keys():
            wordKey = self.wordKeyByIndex[index]
            word = self.summaryWords[wordKey]
            vec = vectors[index]
            if len(word['groups']) == 1:
                cluster = str(word['groups'][0])
            else:
                cluster = '-'.join(str(x) for x in word['groups'])
                
            outV.write('\t'.join([str(x) for x in vec]) + "\n")
            outM.write(word['pure_word'] + "\t" + cluster + "\n")	
            
        outV.close()
        outM.close()
        print('Word vector saved')
        return
    
    def __sortSummaryWords(self, attribute = 'number_of_blocks'):
        if not len(self.summaryWords):
            return
        sortedVocab = {}
        #print(self.summaryWords);
        for value in sorted(self.summaryWords.values(), key=operator.itemgetter(attribute), reverse=True):
            sortedVocab[value['stemmed_word']] = value
        
        self.summaryWords = sortedVocab
        #print(self.summaryWords)
        return
    
    def __saveSummayVocab(self):
        self._saveInPickel(self.__getVocabPath(), self.summaryWords)
        return
    
    def __getVocabPath(self):
        return self._getFilePath(self.prefix + '_meta_vocab.sav', self.path)
    
    def __saveDocs(self):
        self._saveInPickel(self.__getSummaryDocsPath(), self.docsSummaryFile)
        return
    
    def __getSummaryDocsPath(self):
    	return self._getFilePath(self.prefix + '_docs_summary.sav', self.path)
    
    def __getWords(self, text):
        words = word_tokenize(text)
        return pos_tag(words)
    
    def __cleanWord(self, word):
    	return re.sub('[^a-zA-Z0-9]+', '', word)
