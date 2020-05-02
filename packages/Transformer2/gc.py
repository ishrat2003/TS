import corpus
import os, logging, sys
from params.core import Core as Params
from dataset.core import Core as Dataset
import utility

logging.basicConfig(level=logging.INFO)

logging.info("# This script produces the CWR of GC of the given corpus  ")
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

logging.info("# 3. Generating points")
logging.info("# ================================")
cwrProcessor = corpus.CWR(datasetToProcess, params)
cwrProcessor.remove()
# cwrProcessor.removeCategorizedFiles(['Topic 1', 'Topic 2', 'Topic 3', 'Topic 4', 'Topic 5', 'Topic 6'])

points = cwrProcessor.process()
logging.info('Total points: ' + str(len(points)))

logging.info("# 3.1 Generating topic points")
logging.info("# ================================")
cwrProcessor.generateTopicFiles()

logging.info("# 3.2 Generating sentiment points")
logging.info("# ================================")
# cwrProcessor.removeCategorizedFiles(['normal', 'positive', 'negative'])
cwrProcessor.generateSentimentFiles()
             
logging.info("# 4. Display points")
logging.info("# ================================")
imageDirectory = os.path.join(params.plot_directory, params.dataset_name)
plotter = corpus.Plotter(points)

filePath = os.path.join(imageDirectory, 'cwr_gc_plot.png')
file = utility.File(filePath)
file.remove()
plotter.displayPlot(filePath)

logging.info("# 5. Generating word co-occurance")
logging.info("# ================================")
wcoProcessor = corpus.WordCooccurance(datasetToProcess, params)
wcoProcessor.remove()
wcoProcessor.process()

logging.info("# 6. Generating hierarchy")
logging.info("# ================================")
hierarchy = corpus.Hierarchy(datasetToProcess, params)
hierarchy.process()

logging.info("# 6. Generating topic stream data")
logging.info("# ================================")
topicStream = corpus.Topic(datasetToProcess, params)
topicStream.process()

print('Finished')


