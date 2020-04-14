from dataset.core import Core as Dataset
from topic.lda import LDA
from params.core import Core as Params
import os

scriptParams = Params()
params = scriptParams.get()
scriptParams.save(params.data_directory)

dataset = Dataset(params.dataset_name, params.data_directory)

dataProcessor = dataset.get()
lda = LDA(dataProcessor, os.path.join(params.data_directory, params.dataset_name))
lda.setNumberOfTopics(5)
lda.setPerplexity(5)
lda.setNumberOfTopics(5)
lda.setNumberOfTotalTopFrequencyWord(1000)
lda.setNumberOfIterations(1000)
lda.train()
lda.printTopics()