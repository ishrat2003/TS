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
wordProcessor = corpus.RelativeCWR(datasetToProcess, params)
points = wordProcessor.getContext(params.word)
print(points)

logging.info("# 4. Display points")
logging.info("# ================================")
imageDirectory = os.path.join(params.plot_directory, params.dataset_name)
plotter = corpus.Plotter(points)
filePath = os.path.join(imageDirectory, params.word + '_cwr_gc_plot.png')
file = utility.File(filePath)
file.remove()
plotter.displayPlot(filePath)
print('Finished')


