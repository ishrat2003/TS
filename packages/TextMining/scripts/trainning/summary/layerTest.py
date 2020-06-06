import sys
packagesPath = "/content/drive/My Drive/Colab Notebooks/packages/TextMining"
sys.path.append(packagesPath)

import tensorflow as tf
from layers.multiHead import MultiHead as MultiHeadAttention
from layers.encoder import Encoder as EncoderLayer
from layers.decoder import Decoder as DecoderLayer
from layers.sequenceEncoder import SequenceEncoder
from layers.sequenceDecoder import SequenceDecoder
from layers.transformer import Transformer
import numpy as np

temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
out, attn = temp_mha(y, k=y, q=y, mask=None)
print(out.shape)  # (batch_size, encoder_sequence, d_model)
print(attn.shape)  # (batch_size, num_heads, encoder_sequence, encoder_sequence)


sample_encoder_layer = EncoderLayer(512, 8, 2048)

sample_encoder_layer_output = sample_encoder_layer(
    tf.random.uniform((64, 43, 512)), False, None)

print('encoder', sample_encoder_layer_output.shape)  # (batch_size, input_seq_len, d_model)
print('--------------------------------------')

sample_decoder_layer = DecoderLayer(512, 8, 2048)

sample_decoder_layer_output, _, _ = sample_decoder_layer(
    tf.random.uniform((64, 50, 512)), sample_encoder_layer_output, 
    False, None, None)

print(sample_decoder_layer_output.shape)  # (batch_size, target_seq_len, d_model)
print('--------------------------------------')

sample_encoder = SequenceEncoder(num_layers=2, d_model=512, num_heads=8, 
                         dff=2048, input_vocab_size=8500,
                         maximum_position_encoding=10000)
temp_input = tf.random.uniform((64, 62), dtype=tf.int64, minval=0, maxval=200)

sample_encoder_output = sample_encoder(temp_input, training=False, mask=None)

print (sample_encoder_output.shape)  # (batch_size, input_seq_len, d_model)

print('--------------------------------------')
sample_decoder = SequenceDecoder(num_layers=2, d_model=512, num_heads=8, 
                         dff=2048, target_vocab_size=8000,
                         maximum_position_encoding=5000)
temp_input = tf.random.uniform((64, 26), dtype=tf.int64, minval=0, maxval=200)

output, attn = sample_decoder(temp_input, 
                              encoderOutput=sample_encoder_output, 
                              training=False,
                              lookAheadMask=None, 
                              paddingMask=None)

print(output.shape, attn['decoder_layer2_block2'].shape)
print('--------------------------------------')

sample_transformer = Transformer(
    num_layers=2, d_model=512, num_heads=8, dff=2048, 
    input_vocab_size=8500, target_vocab_size=8000, 
    pe_input=10000, pe_target=6000)

temp_input = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)
temp_target = tf.random.uniform((64, 36), dtype=tf.int64, minval=0, maxval=200)

fn_out, _ = sample_transformer(temp_input, 
                              temp_target, training=False, 
                               encoderPaddingMask=None, 
                               decoderTargetPaddingAndLookAheadMask=None,
                               decoderPaddingMask=None)
print(fn_out.shape)  # (batch_size, tar_seq_len, target_vocab_size)

print('--------------------------------------')

