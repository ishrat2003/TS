import tensorflow as tf
from layers.transformer import Transformer
from layers.customScheduler import CustomScheduler
from layers.mask import Mask
from layers.loss import Loss
import time
import numpy
import  sys

class Trainer:
    
    def __init__(self, params):
        self.params = params
        self.dModel = self.params.d_model
        self.optimizer = self.getOptimizer()
        self.model = None
        self.mask = Mask()
        self.loss = Loss()
        self.setLossMetrics()
        self.setAccuracyMetrics()
        return
    
    def getOptimizer(self):
        learningRate = CustomScheduler(self.dModel)
        return tf.optimizers.Adam(learningRate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    
    def setLossMetrics(self):
        self.trainLoss = tf.keras.metrics.Mean(name='train_loss')
        return
    
    def setAccuracyMetrics(self):
        self.trainAccuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        return

    def setModel(self, model):
        self.model = model
        return
    
    def setCheckpoint(self, checkpointPath):
        self.ckpt = tf.train.Checkpoint(transformer=self.model, optimizer=self.optimizer)
        self.ckptManager = tf.train.CheckpointManager(self.ckpt, checkpointPath, max_to_keep=10)

        # if a checkpoint exists, restore the latest checkpoint. https://www.tensorflow.org/guide/checkpoint#loading_mechanics
        if self.ckptManager.latest_checkpoint:
            self.ckpt.restore(self.ckptManager.latest_checkpoint)
            print ('Latest checkpoint restored!!')
            
        return
    
    def process(self, epochs, dataset):
        # The @tf.function trace-compiles train_step into a TF graph for faster
        # execution. The function specializes to the precise shape of the argument
        # tensors. To avoid re-tracing due to the variable sequence lengths or variable
        # batch sizes (the last batch is smaller), use input_signature to specify
        # more generic shapes.

        trainStepSignature = [
            tf.TensorSpec(shape=(None, None), dtype=tf.int64),
            tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        ]
        # @tf.function(input_signature=trainStepSignature)
        def trainStep(source, target):
            # tf print doesn't work in jupyter
            # tf.print(source, output_stream=sys.stdout)
            targetInput = target[:, :-1]
            targetReal = target[:, 1:]

            encoderPaddingMask, decoderTargetPaddingAndLookAheadMask, decoderPaddingMask = self.mask.createMasks(source, target)

            if (self.params.display_details == True) :
                print('trainStep target: ', target.shape)
                print('trainStep targetInput: ', targetInput.shape)
                print('trainStep targetReal: ', targetReal.shape)
                print('trainStep encoderPaddingMask', encoderPaddingMask.shape)
                print('trainStep decoderTargetPaddingAndLookAheadMask', decoderTargetPaddingAndLookAheadMask.shape)
                print('trainStep decoderPaddingMask', decoderPaddingMask.shape)
                
            with tf.GradientTape() as tape:
                if (self.params.display_details == True) :
                    print('G tape, source', source.shape, tf.shape(source))
                    print('G tape, targetInput', targetInput.shape, tf.shape(targetInput))
                predictions, _ = self.model(source, 
                    targetInput, 
                    True, 
                    encoderPaddingMask, 
                    decoderTargetPaddingAndLookAheadMask, 
                    decoderPaddingMask)
                    
                loss = self.loss.lossFunction(targetReal, predictions)

            gradients = tape.gradient(loss, self.model.trainable_variables)    
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            self.trainLoss(loss)
            self.trainAccuracy(targetReal, predictions)
            return
    
        for epoch in range(epochs):
            self.startEpoch(epoch)

            # inp -> portuguese, tar -> english
            for (batch, (source, target)) in enumerate(dataset):
                if (self.params.display_details == True) :
                    print('Batch: ', batch)
                    print('source', source.shape)
                    print('target', target.shape)
                trainStep(source, target)
                self.endBatch(batch, epoch)

            self.endEpoch(batch, epoch)
                
        return
    
    def startEpoch(self, epoch):
        print('Starting epoch: ', (epoch+1))
        self.start = time.time()
        self.trainLoss.reset_states()
        self.trainAccuracy.reset_states()
        return
    
    def endBatch(self, batch, epoch):
        if batch % 50 == 0:
            print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, batch, self.trainLoss.result(), self.trainAccuracy.result()))
        return

    def endEpoch(self, batch, epoch):
        ckptSavePath = self.ckptManager.save()
        print ('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckptSavePath))

        print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, self.trainLoss.result(), self.trainAccuracy.result()))
        print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - self.start))
        return