from .base import Base
import tensorflow_datasets as tfds

class CNNDailyMail(Base):
    
    def __init__(self, path):
        super().__init__(path, 'cnn_dailymail')
        return

    def dataset(self):
        data, metadata = tfds.load('cnn_dailymail', 
            data_dir = self.getProcessedPath(), 
            with_info = True, 
            as_supervised = True, 
            download_and_prepare_kwargs = dict(download_config=self.dlConfig))
        
        return data['train'], data['validation']
    
    def getGenerator(self, trainingSet, type = 'source'):
        if type == 'source':
            return (article.numpy() for article, highlight in trainingSet)
        
        return (highlight.numpy() for article, highlight in trainingSet)
    
