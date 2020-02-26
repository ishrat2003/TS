import argparse
import os
import logging
import json

class Core:

  def __init__(self):
    self.parser = argparse.ArgumentParser()
    self.parser.add_argument('--display_details', default=False, type=bool)
    self.parser.add_argument('--display_network', default=True, type=bool)
    self.parser.add_argument('--dataset_name', default='newsroom')
    self.parser.add_argument('--data_directory', default='/content/drive/My Drive/Colab Notebooks/data')
    
    self.parser.add_argument('--source_max_sequence_length', default=1000) # Content length
    self.parser.add_argument('--target_max_sequence_length', default=300) # Summary length
    
    self.parser.add_argument('--buffer_size', default=20000)
    self.parser.add_argument('--batch_size', default=8)
    self.parser.add_argument('--dimensions', default=512)
    self.parser.add_argument('--num_layers', default=4)
    self.parser.add_argument('--num_heads', default=8)
    self.parser.add_argument('--d_model', default=128)
    self.parser.add_argument('--dff', default=512, help = "Positive integer, dimensionality of the output space for FF network")
    self.parser.add_argument('--dropout_rate', default=0.1)
    
    self.parser.add_argument('--checkpoint_directory', default='/content/drive/My Drive/Colab Notebooks/data/checkpoints')
    self.parser.add_argument('--log_directory', default='/content/drive/My Drive/Colab Notebooks/data/logs')
    self.parser.add_argument('--plot_directory', default='/content/drive/My Drive/Colab Notebooks/data/plots')
    return

  def get(self):
    return self.parser.parse_args()

  def save(self, path):
    if not os.path.exists(path):
      os.makedirs(path)

    params = json.dumps(vars(self.get()))
    path = os.path.join(path, self._getFileName())
    with open(path, 'w') as fileToProcess:
        fileToProcess.write(params)

    logging.info("# Params saved in " + path)
    return

  def load(self, path):
    if not os.path.isdir(path):
        path = os.path.dirname(path)

    path = os.path.join(path, self._getFileName())
    fileContent = open(path, 'r').read()
    flag2val = json.loads(fileContent)
    for flag, value in flag2val.items():
        self.parser.flag = value

  def _getFileName(self):
    return type(self).__name__ + "params"