import numpy as np
import utility

# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

class Store():    
    
    def _loadNumpy(self, fileName, path):
        filePath = self._getFilePath(fileName, path)
        file = utility.File(filePath)
        if not file.exists():
            return None

        return np.load(filePath)


    def _saveNumpy(self, fileName, data, path):
        filePath = self._getFilePath(fileName, path)
        file = utility.File(filePath)
        file.remove()
        np.savez(filePath, data)
        return


    def _getFilePath(self, fileName, path):
        return utility.File.join(path, fileName)