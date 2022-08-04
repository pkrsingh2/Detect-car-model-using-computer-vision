# -------------------------------------------------------------------------------------------------------------------------------
import tensorflow as tf
# customised learning rate scheduler
# considering slope of validation loss
class lossGradientLR(tf.keras.callbacks.Callback):
    """gradually descend learning rate if the loss defined my the model is not descending at desired rate"""
    def __init__(self,initial_learning_rate,patience=5,
                 slope=0.25,factor=0.75,lr_least=1e-10,verbose=False):
        """arguments:
        ----------
        initial_learning_rate : the starting value of learning rate
        patience : minimum number of epochs before manipulation
        slope : loss gradient descent threshold
        factor : multiplying factor for learning rate change
        lr_least : least value of learning rate to apply
        verbose : status message display control flag
        """
        super(lossGradientLR, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.patience = patience
        self.slope = slope
        self.factor = factor
        self.lr_least = lr_least
        self.verbose = verbose
        self.loss = []
        self.lr = [initial_learning_rate]
        self.lastAdjust = 0
        
    def on_epoch_end(self,epoch,logs={}):
        """review learning rate & calculate
        arguments:
        ----------
        epoch : number of the epoch that ended
        logs : learning evaluation logs
        """
        # fetch validation loss
        self.loss.append(logs.get('val_loss'))
        
        from tensorflow.keras import backend as K
        
        if self.lr[-1]<=self.lr_least: # no updates if at lr_least
            return        
        
        # checkpoints
        flag1 = (epoch-self.lastAdjust)>self.patience # min epoch before adjust
        try:
            flag2 = (self.loss[-1]-self.loss[-2])>0 and (self.loss[-2]-self.loss[-3])>0
            # if two continuous epochs have increasing validation loss
        except:
            flag2 = False # fails in first 4 epochs, handle it
        
        # slope of validation loss
        lossGrad = self.loss[-self.patience:][0] - self.loss[-self.patience:][-1]
        lossGrad /= self.loss[-self.patience:][0]
        
        # if slope of validation loss not descending as expected by slope
        # or flag2 : if two continuous epochs have increasing validation loss
        if (flag1 and lossGrad<self.slope) or (flag2):
            self.lr.append(max(self.lr[-1]*self.factor,self.lr_least)) #min lr_least
            self.lastAdjust = epoch # update last adjust epoch
            
        # update model learning rate
        K.set_value(self.model.optimizer.learning_rate, self.lr[-1])
        
        if self.verbose:
            print("\n\n-------------------------------------------------------")
            print("slopeOfLoss:%.2f"%lossGrad," @ epoch:%3d"%epoch, "---- learning_rate:",self.lr[-1])
            print("-------------------------------------------------------\n")
            
        super().on_epoch_end(epoch, logs)
        
# report learning rate to tensorboard
# due to custom lr_scheduler
class LRTensorBoard(tf.keras.callbacks.TensorBoard):
    """report to tensorfboard about the updates created by lossGradientLR"""
    def __init__(self, log_dir, **kwargs):
        super(LRTensorBoard,self).__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        from tensorflow.keras import backend as K
        logs = logs or {}
        logs.update({'learning_rate': K.eval(self.model.optimizer.learning_rate)})
        super().on_epoch_end(epoch, logs)
# -------------------------------------------------------------------------------------------------------------------------------
class eStop(tf.keras.callbacks.Callback):
    """stop training if gradient descent is ineffective over few epochs"""
    def __init__(self,patience=11,slope=0.25):
        """arguments:
        ----------
        patience : Number of epochs with no improvement after which training will be stopped
        slope = loss gradient descent threshold
        """
        super(eStop,self).__init__()
        self.patience = patience
        self.slope = slope
        self.loss = []
        
    def on_epoch_end(self,epoch,logs={}):
        """review losses, decide EarlyStop implementaion
        arguments:
        ----------
        epoch : number of the epoch that ended
        logs : learning evaluation logs
        """
        
        # fetch validation loss
        self.loss.append(logs.get('val_loss'))
        
        # allow minimum number of epoch
        if len(self.loss)<self.patience:
            return
        
        # compute slope of loss
        lossGrad = self.loss[-self.patience:][0] - self.loss[-self.patience:][-1]
        lossGrad /= self.loss[-self.patience:][0]
        if lossGrad<self.slope: # if threshold attained
            self.model.stop_training = True # stop training
            print("\n\n---------------------------------------------------------------")
            print("eStop initiated after epoch %3d for poor loss gradient of %.3f"%(epoch,lossGrad))
            print("---------------------------------------------------------------\n")
            return
        
        super().on_epoch_end(epoch, logs)
# -------------------------------------------------------------------------------------------------------------------------------
class layerUnFreeze(tf.keras.callbacks.Callback):
    """control layer trainable status over the epoch"""
    def __init__(self,uncontrolled=-5,schedule={}):
        """arguments:
        ----------
        uncontrolled : number of layers (from the output end) to be left trainable always eg. -15 or -5 from output
        schedule : dictionary of % of layers as values mapped to epochs as keys in ascending order
        this defines the % of layers to be open for training from the respective epochs
        this affects only the layes not marked by 'uncontrolled'
        eg. {20:0.25, 25:0.5, 30:0.75, 35:0.9}
        """
        super(layerUnFreeze,self).__init__()
        self.uncontrolled = uncontrolled
        self.schedule = schedule
        
    def on_train_begin(self,logs=None):
        """preset the layer trainable layers"""
        self.total = len(self.model.layers)
        for i,layer in enumerate(self.model.layers):
            if i < self.total+self.uncontrolled:
                layer.trainable = False # controlled layers
            else:
                layer.trainable = True # uncontrolled layers
                
        # schedule of layers to unfreeze
        import numpy as np
        epochs = np.arange(1,list(self.schedule.keys())[-1]+1)
        self.trainables = np.round(np.array(list(map(self.schedule.get,
                                                     epochs,[0]*len(epochs))))*(self.total+self.uncontrolled)).astype(int)        
        for i in range(1,len(self.trainables)):
            if self.trainables[i]==0:
                self.trainables[i]=self.trainables[i-1]
        self.trainables = -self.trainables+self.uncontrolled
        
        # invoke super method
        super().on_train_begin(logs)
        
    def on_epoch_end(self,epoch,logs=None):
        """unfreeze layers as per schedule"""
        if epoch>=len(self.trainables):
            unfreeze = self.trainables[-1]
        else:
            unfreeze = self.trainables[epoch]
        for layer in self.model.layers[unfreeze:]:
            layer.trainable = True #unfreezed
        super().on_epoch_end(epoch, logs)
        
# report trainables to tensorboard
class trainableReport(tf.keras.callbacks.TensorBoard):
    """report to tensorboard about the updates created by layerUnFreeze"""
    def __init__(self, log_dir, **kwargs):
        super(trainableReport,self).__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        trainable=0
        for layer in self.model.layers:
            trainable+= layer.trainable
        logs.update({'Trainable': trainable})
        super().on_epoch_end(epoch, logs)
# -------------------------------------------------------------------------------------------------------------------------------
from timeit import default_timer as timer
class timeLog(tf.keras.callbacks.TensorBoard):
    """to log epoch times as a measure of computational resource consumption"""
    
    def __init__(self, log_dir, **kwargs):
        super(timeLog,self).__init__(log_dir=log_dir, **kwargs)
        
    def on_epoch_begin(self, epoch, logs={}):
        self.tBegin = timer()        
        super().on_epoch_begin(epoch, logs)
        
    def on_epoch_end(self, epoch, logs={}):
        logs = logs or {}
        logs.update({'lapTime':(timer()-self.tBegin)})
        super().on_epoch_end(epoch, logs)
