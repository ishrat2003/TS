import lc as LC
import os, logging, sys
from params.core import Core as Params
from dataset.core import Core as Dataset

logging.basicConfig(level=logging.INFO)

logging.info("# 1. Loading script params ")
logging.info("# ================================")
scriptParams = Params()
params = scriptParams.get()
scriptParams.save(params.data_directory)

logging.info("# 2. Preprocessing data")
logging.info("# ================================")

dataset = Dataset(params.dataset_name, params.data_directory)
datasetToProcess = dataset.get(float(params.dataset_percentage), int(params.total_items))

if not datasetToProcess:
    logging.error('No dataset found')
    sys.exit()

data = datasetToProcess.getTrainingSet()

imageDirectory = os.path.join(params.plot_directory, params.dataset_name)

for item in data.take(1):
    label = datasetToProcess.getLabel(item)
    sourceText = datasetToProcess.getText(item)
    print('label::::::: ', label)
    print('title::::::: ', datasetToProcess.getTitle(item))
    print('abstract::::::: ', datasetToProcess.getAbstract(item))
    filePrefix = 'test'
    allowedTypes = ['NN', 'NNP', 'NNS', 'NNPS']
    
    print('CWR ::::: ')
    peripheralProcessor = LC.Peripheral(sourceText)
    peripheralProcessor.setAllowedPosTypes(allowedTypes)
    peripheralProcessor.setPositionContributingFactor(5)
    peripheralProcessor.setOccuranceContributingFactor(0)
    peripheralProcessor.setProperNounContributingFactor(0)
    peripheralProcessor.setTopScorePercentage(0.2)
    peripheralProcessor.setFilterWords(0)
    peripheralProcessor.loadSentences(sourceText)
    peripheralProcessor.loadFilteredWords()
    peripheralProcessor.train()
    peripheralProcessor.getPoints()
    peripheralProcessor.displayPlot(os.path.join(imageDirectory, filePrefix + '_peripheral.png'))
    contributors = peripheralProcessor.getContrinutors()
    print('contributors', contributors)

    print('TSNE ::::: ')
    tsneProcessor = LC.TSNELC(sourceText)
    tsneProcessor.setAllowedPosTypes(allowedTypes)
    tsneProcessor.setPerplexity(3)
    tsneProcessor.setNumberOfComponents(2)
    tsneProcessor.setNumberOfIterations(1000)
    tsneProcessor.setTopScorePercentage(0.2)
    tsneProcessor.setFilterWords(0)
    tsneProcessor.loadSentences(sourceText)
    tsneProcessor.loadFilteredWords()
    tsneProcessor.setMarkedWords(contributors)
    tsneProcessor.train(imageDirectory)
    print('word info ::::: ')
    print(tsneProcessor.getWordInfo())

    
    print('display tsne')
    tsneProcessor.displayPlot(os.path.join(imageDirectory, filePrefix + '_tsne.png'))

    print('Linear ::::: ')
    linearProcessor = LC.Linear(sourceText)
    linearProcessor.setAllowedPosTypes(allowedTypes)
    linearProcessor.setPositionContributingFactor(5)
    linearProcessor.setOccuranceContributingFactor(0)
    linearProcessor.setProperNounContributingFactor(0)
    linearProcessor.setTopScorePercentage(0.2)
    linearProcessor.setFilterWords(0)
    linearProcessor.loadSentences(sourceText)
    linearProcessor.loadFilteredWords()
    linearProcessor.train()
    
    print('display linear')
    linearProcessor.displayPlot(os.path.join(imageDirectory, filePrefix + '_linear.png'))

print('Finished')


