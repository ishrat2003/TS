import numpy as np
import utility

# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

class Store():    
    

    def __init__(self, dataset, path):
        self.dataset = dataset
        self.path = path
        self.prefix = ''
        return


    def setPrefix(self, prefix):
        self.prefix = prefix
        return


    def _loadNumpy(self, fileName):
        filePath = self._getFilePath(fileName)
        file = utility.File(filePath)
        if not file.exists():
            return None

        return np.load(filePath)


    def _saveNumpy(self, fileName, data):
        filePath = self._getFilePath(fileName)
        file = utility.File(filePath)
        file.remove()
        np.savez(filePath, data)
        return


    def _getFilePath(self, fileName):
        fileName = self.prefix + fileName
        return utility.File.join(self.path, fileName)