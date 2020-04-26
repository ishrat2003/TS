import corpus
import os, logging, sys
from params.core import Core as Params
from dataset.core import Core as Dataset

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

# logging.info("# 3. Generating points")
# logging.info("# ================================")
# cwrProcessor = corpus.CWR(datasetToProcess, params)
# points = cwrProcessor.process()
# logging.info('Total points: ', len(points))
             
# logging.info("# 4. Display points")
# logging.info("# ================================")
# imageDirectory = os.path.join(params.plot_directory, params.dataset_name)
# plotter = corpus.Plotter(points)
# plotter.displayPlot(os.path.join(imageDirectory, 'cwr_gc_plot.png'))

logging.info("# 5. Generating word co-occurance")
logging.info("# ================================")

wcoProcessor = corpus.WordCooccurance(datasetToProcess, params)
wcoProcessor.process()

print('Finished')


