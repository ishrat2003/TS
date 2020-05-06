from dataset.core import Core as Dataset
from topic.lda import LDA
from topic.evaluate import Evaluate
from params.core import Core as Params
import os

scriptParams = Params()
params = scriptParams.get()
scriptParams.save(params.data_directory)

dataset = Dataset(params.dataset_name, params.data_directory)

dataProcessor = dataset.get()
if params.type == 'lda':
    print(":::::::::::::: Evaluating ::::::::::::::")
    evaluationProcessor = Evaluate(dataProcessor, params)
    evaluationProcessor.process()
else:
    print(":::::::::::::: Train ::::::::::::::")
    lda = LDA(dataProcessor, os.path.join(params.data_directory, params.dataset_name))
    lda.remove()
    lda.setPerplexity(5)
    lda.setNumberOfTopics(6)
    lda.setNumberOfTotalTopFrequencyWord(10000)
    lda.setNumberOfIterations(1000)
    lda.train()
    lda.printTopics()
    
print(":::::::::::::: Finished ::::::::::::::")