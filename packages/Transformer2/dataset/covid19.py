import pathlib
import tensorflow as tf
import json
import lc
from .base import Base
import re, sys, numpy
from nltk import word_tokenize, pos_tag
from nltk.stem.porter import PorterStemmer
import utility
from datetime import datetime

class Covid19(Base):
    
    def __init__(self, path):
        super().__init__(path, 'covid19')
        self.allowedPOSTypes = ['NN', 'NNP', 'NNS', 'NNPS']
        self.minCount = 3
        self.stopWords = utility.Utility.getStopWords()
        self.stemmer = PorterStemmer()
        return
    
    def setAllowedPosTypes(self, allowedPOSTypes):
        self.allowedPOSTypes = allowedPOSTypes
        return

    def get(self):
        # datetime object containing current date and time
        now = datetime.now()
        
        print("now =", now)
        articlesRoot = tf.keras.utils.get_file(self.directoryPath + '/covid19/biorxiv_medrxiv', 
            'https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-03-27/biorxiv_medrxiv.tar.gz',
            untar=True)

        articlesRoot = pathlib.Path(articlesRoot)
        filePaths = tf.data.Dataset.list_files(str(articlesRoot/'*'))
        
        def read(path):
            label = tf.strings.split(path, '/')[-1]
            return tf.io.read_file(path), label
        
        dataset = filePaths.map(read)
        return dataset
    
    def getText(self, rawData, withTitle = False):
        data = self.__getData(rawData)
        abstractText = [paragraph["text"] for paragraph in data["abstract"]]
        bodyText = [paragraph["text"] for paragraph in data["body_text"]]
        text = ' '.join(abstractText) + ' '.join(bodyText)
        if withTitle:
            text = data["metadata"]["title"] + '. ' + text
        return self.__getProcessedText(text)
    
    def getTitle(self, rawData):
        data = self.__getData(rawData)
        return self.__getProcessedText(data["metadata"]["title"])
    
    def getAbstract(self, rawData):
        data = self.__getData(rawData)
        return self.__getProcessedText(data["metadata"]["abstract"])
    
    # def getLCTWordsOccurredMoreThanMinCount(self, item):
    #     itemData, _ = item
    #     text = self.getText(itemData.numpy())
        
    #     lcProcessor = lc.Peripheral(text, 0)
    #     lcProcessor.setAllowedPosTypes(self.allowedPOSTypes)
    #     lcProcessor.setFilterWords(0)
    #     lcProcessor.train()
        
    #     localWords = lcProcessor.getWordInfo()
    #     words = []
        
    #     for word in localWords:
    #         if localWords[word]['count'] <= self.minCount:
    #             continue
    #         words.append(localWords[word]['stemmed_word'])
   
    #     return ' '.join(words)
    
    def __getData(self, rawData):
        return json.loads(rawData.decode("utf-8"))

    def __getProcessedText(self, text):
        words = word_tokenize(self.__clean(text))
        allWords = pos_tag(words)
        processedWords = []

        for itemWord in allWords:
            (word, type) = itemWord
            if (type not in self.allowedPOSTypes):
                continue

            word = self.__cleanWord(word)
            word = self.stemmer.stem(word.lower())
            processedWords.append(word)

        return ' '.join(processedWords)

    def __cleanWord(self, word):
        return re.sub('[^a-zA-Z0-9]+', '', word)
    
    def __clean(self, text):
        text = re.sub('<.+?>', '. ', text)
        text = re.sub('&.+?;', '', text)
        text = re.sub('[\']{1}', '', text)
        text = re.sub('[^a-zA-Z0-9\s_\-\?:;\.,!\(\)\"]+', ' ', text)
        text = re.sub('\s+', ' ', text)
        text = re.sub('(\.\s*)+', '. ', text)
        return text
