import pathlib
import tensorflow as tf
import json
import lc

class PharmaNews():
    
    def __init__(self, path):
        self.directoryPath = path
        self.name = 'covid19'
        self.allowedPOSTypes = ['NN', 'NNP', 'NNS', 'NNPS']
        self.minCount = 3
        return

    def get(self):
        articlesRoot = tf.keras.utils.get_file(self.directoryPath + '/biorxiv_medrxiv', 
            'https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-03-27/biorxiv_medrxiv.tar.gz',
            untar=True)

        articlesRoot = pathlib.Path(articlesRoot)
        filePaths = tf.data.Dataset.list_files(str(articlesRoot/'*'))
        
        def read(path):
            label = tf.strings.split(path, '/')[-2]
            return tf.io.read_file(path), label
        
        dataset = filePaths.map(read)
        return dataset
    
    def getText(self, rawData):
        data = json.loads(rawData.decode("utf-8"))
        abstractText = [paragraph["text"] for paragraph in data["abstract"]]
        bodyText = [paragraph["text"] for paragraph in data["body_text"]]
        text = data["metadata"]["title"] + '. ' + ' '.join(abstractText) + ' '.join(bodyText)
        return text
    
    def getLCTWordsOccurredMoreThanMinCount(self, item):
        itemData, _ = item
        text = self.getText(itemData.numpy())
        
        lcProcessor = lc.Peripheral(text, 0)
        lcProcessor.setAllowedPosTypes(self.allowedPOSTypes)
        lcProcessor.setFilterWords(0)
        lcProcessor.train()
        
        localWords = lcProcessor.getWordInfo()
        words = []
        
        for word in localWords:
            if localWords[word]['count'] <= self.minCount:
                continue
            words.append(localWords[word]['stemmed_word'])
   
        return ' '.join(words)

    # def getGenerator(self, trainingSet, type = 'source'):
    #     if type == 'source':
    #         return (article.numpy() for article, highlight in trainingSet)
        
    #     return (highlight.numpy() for article, highlight in trainingSet)
