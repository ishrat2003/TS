import sys
packagesPath = "/content/drive/My Drive/Colab Notebooks/packages/TextMining"
sys.path.append(packagesPath)

import tensorflow as tf
from layers.scaledDot import ScaledDot
import numpy as np

np.set_printoptions(suppress=True)

k = tf.constant([[10,0,0],
                [0,10,0],
                [0,0,10],
                [0,0,10]], dtype=tf.float32)  # (4, 3)

v = tf.constant([[   1,0],
                      [  10,0],
                      [ 100,5],
                      [1000,6]], dtype=tf.float32)  # (4, 2)

# This `query` aligns with the second `key`,
# so the second `value` is returned.
q = tf.constant([[0, 10, 0]], dtype=tf.float32)  # (1, 3)

mask = tf.constant([[1,1,1,0]], dtype=tf.float32)  # (1, 4)
# q = n x seq_length
# k = d1 x seq_length
# v = d1 x m 
# mask = n x d1
# attention = n x seq_length
# output = n x m

attentionSDP = ScaledDot()
temp_out, temp_attn = attentionSDP.attention(q, k, v, None)
print ('Attention weights are:')
print (temp_attn)
print ('Output is:')
print (temp_out)


temp_out, temp_attn = attentionSDP.attention(q, k, v, mask)
print ('Attention weights are:')
print (temp_attn)
print ('Output is:')
print (temp_out)