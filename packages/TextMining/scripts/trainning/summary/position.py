from __future__ import absolute_import, division, print_function, unicode_literals

import sys
packagesPath = "/content/drive/My Drive/Colab Notebooks/packages/TextMining"
sys.path.append(packagesPath)

import tensorflow_datasets as tfds
import tensorflow as tf
import os, logging, sys
import time
import numpy as np
import matplotlib.pyplot as plt

from params.core import Core as Params
from layers.positionalEncoding import PositionalEncoding

logging.basicConfig(level=logging.INFO)
scriptParams = Params()
params = scriptParams.get()

positionalEncoding = PositionalEncoding(params.source_max_sequence_length, params.dimensions) 
positionalEmbedding = positionalEncoding.getEmbedding()
print (positionalEmbedding.shape)

plt.pcolormesh(positionalEmbedding[0], cmap='RdBu')
plt.xlabel('Depth')
plt.xlim((0, 512))
plt.ylabel('Position')
plt.colorbar()
plt.show()