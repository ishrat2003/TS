from .meta import Meta
from .visualization import Visualization
import numpy
import io, os
import utility

class WordCooccurance(Visualization):
    
    def __init__(self, datasetProcessor, params):
        super().__init__(datasetProcessor, params)
        
        meta = Meta(self.datasetProcessor)
        self.docsLCs = meta.loadDocLcs()
        return
    
    def getPoints(self):
        topics = self.filteredWordsByTopic.keys()
        totalTopics = len(topics)
        
        self.points = {}
        self.wordIds = {}
        
        for topic in topics:
            self.appendPoints(self.filteredWordsByTopic[topic], topic)

        totalPoints = len(self.points)
        vectors = numpy.zeros((totalPoints, totalPoints))

        for doc in self.docsLCs.keys():
            relatedIndecies = self.docsLCs[doc]
            for wordIndex1 in relatedIndecies:
                if wordIndex1 not in self.wordIds.keys():
                    continue
                row = self.wordIds[wordIndex1]
                for wordIndex2 in doc:
                    if wordIndex2 not in self.wordIds.keys():
                        continue
                    column = self.wordIds[wordIndex2]
                    vectors[row][column] += 1
                    
        self.saveVectorAndMeta(vectors, self.points)
            
        return vectors
    
    def appendPoints(self, words, topic):
        row = 0
        for word in words:
            self.wordIds[word['index']] = row
            self.points[word['index']] = word
            row += 1

        return
    
    def saveVectorAndMeta(self, vectors, points):
        outV = io.open(os.path.join(self.path, 'embedding_vecs.tsv'), 'w', encoding='utf-8')
        outM = io.open(os.path.join(self.path, 'embedding_meta.tsv'), 'w', encoding='utf-8')

        writeHeader = True
        outM.write("word\ttopic\n")

        for index in self.wordIds.keys():
            row = self.wordIds[index]
            vec = vectors[row]
            outV.write('\t'.join([str(x) for x in vec]) + "\n")
            word = points[index]
            outM.write(word['label'] + "\t" + word['topic'] + "\n")	
            
        outV.close()
        outM.close()
        print('Word vector saved')
        return
    
    def remove(self):
        file = utility.File(os.path.join(self.path, 'embedding_vecs.tsv'))
        file.remove()
        file = utility.File(os.path.join(self.path, 'embedding_meta.tsv'))
        file.remove()
        return