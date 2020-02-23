import os
import tensorflow_datasets as tfds
from .base import Base
from .newsroom import Newsroom
from .ptToEnTranslate import PtToEnTranslate
from .scientificPapers import ScientificPapers
from .cnnDailyMail import CNNDailyMail

class Core:

    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.dataSetProcessor = Base(self.path, self.name)
        return
    
    def get(self):
        if (self.name == 'newsroom'):
            self.dataSetProcessor = Newsroom(self.path)
        elif (self.name == 'pt_to_en_translate'):
            self.dataSetProcessor = PtToEnTranslate(self.path)
        elif (self.name == 'scientific_papers_arxiv'):
            self.dataSetProcessor = ScientificPapers(self.path, 'scientific_papers/arxiv')
        elif (self.name == 'scientific_papers_pubmed'):
            self.dataSetProcessor = ScientificPapers(self.path, 'scientific_papers/pubmed')
        elif (self.name == 'cnn_dailymail'):
            self.dataSetProcessor = CNNDailyMail(self.path)

        return self.dataSetProcessor
    
    
   