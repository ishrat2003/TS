import tensorflow as tf
from layers.transformer import Transformer
from layers.mask import Mask
import matplotlib.pyplot as plt
import os
import datetime

class Sequence():

    def __init__(self, params, sourceTokenizer, targetTokenizer):
        self.maxLength = params.target_max_sequence_length
        self.sourceTokenizer = sourceTokenizer
        self.targetTokenizer = targetTokenizer
        inputVocabSize = self.sourceTokenizer.vocab_size + 2
        targetVocabSize = self.targetTokenizer.vocab_size + 2
        self.model = Transformer(params, inputVocabSize, targetVocabSize, 
                          inputVocabSize, 
                          targetVocabSize)
        self.mask = Mask()
        self.attentionWeights = None
        self.outputSentences = None
        self.plotDir = os.path.join(params.plot_directory, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        return

    def process(self, source):
        startToken = [self.sourceTokenizer.vocab_size]
        endToken = [self.sourceTokenizer.vocab_size + 1]

        sourceInput = startToken + self.sourceTokenizer.encode(source) + endToken
        encoderInput = tf.expand_dims(sourceInput, 0)

        # as the target is english, the first word to the transformer should be the
        # english start token.
        decoderInput = [self.targetTokenizer.vocab_size]
        target = tf.expand_dims(decoderInput, 0)

        for i in range(self.maxLength):
            encoderPaddingMask, decoderTargetPaddingAndLookAheadMask, decoderPaddingMask = self.mask.createMasks(encoderInput, target)
        
            # predictions.shape == (batch_size, seq_len, vocab_size)    
            predictions, attentionWeights = self.model(encoderInput, 
                target,
                False,
                encoderPaddingMask,
                decoderTargetPaddingAndLookAheadMask,
                decoderPaddingMask)
            
            # select the last word from the seq_len dimension
            predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

            
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # return the result if the predicted_id is equal to the end token
            if predicted_id == self.targetTokenizer.vocab_size + 1:
                return tf.squeeze(target, axis=0)

            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            target = tf.concat([target, predicted_id], axis=-1)

        output = tf.squeeze(target, axis=0)
        self.outputSentence = self.targetTokenizer.decode([i for i in output if i < self.targetTokenizer.vocab_size])
        self.attentionWeights = attentionWeights
        
        return self.outputSentence
    
    # def plotAttention(self, attentionWeights, source, outputSentence, layer = 0):
    #     fig = plt.figure(figsize=(16, 8))
    #     sentence = self.sourceTokenizer.encode(sentence)
    #     attention = tf.squeeze(attentionWeights[layer], axis=0)

    #     for head in range(attentionWeights.shape[0]):
    #     ax = fig.add_subplot(2, 4, head+1)

    #     # plot the attention weights
    #     ax.matshow(attentionWeights[head][:-1, :], cmap='viridis')

    #     fontdict = {'fontsize': 10}

    #     ax.set_xticks(range(len(sentence)+2))
    #     ax.set_yticks(range(len(outputSentence)))

    #     ax.set_ylim(len(outputSentence)-1.5, -0.5)

    #     ax.set_xticklabels(
    #     ['<start>']+[self.sourceTokenizer.decode([i]) for i in sentence]+['<end>'], 
    #     fontdict=fontdict, rotation=90)

    #     ax.set_yticklabels([self.targetTokenizer.decode([i]) for i in output 
    #                     if i < self.targetTokenizer.vocab_size], 
    #                     fontdict=fontdict)

    #     ax.set_xlabel('Head {}'.format(head+1))

    #     plt.tight_layout()
    #     plt.show()