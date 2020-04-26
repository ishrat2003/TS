import pickle
import utility
import os

class Writer():
    
    def getFile(self, filename, writeHeader = True):
        path = os.path.join(self.path, filename)
        file = utility.File(path, writeHeader)
        return file
    
    def _getFilePath(self, fileName, path):
        return utility.File.join(path, fileName)

    def _getFromPickel(self, filePath):
        file = utility.File(filePath)
        if file.exists():
            return pickle.load(open(filePath, 'rb'));
        return None

    def _saveInPickel(self, filePath, model):
        file = utility.File(filePath)
        if file.exists():
            file.remove()
        pickle.dump(model, open(filePath, 'wb'))
        return
