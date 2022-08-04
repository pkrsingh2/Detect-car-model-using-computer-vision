# -------------------------------------------------------------------------------------------------------------------------------
import tensorflow as tf
# fetch GPU device name
GPU = tf.config.list_logical_devices('GPU')[0].name

# compute Intersection-over-Union metrics
def IOU(y_true,y_pred):
    """arguments:
    ----------
    y_true : true bounding box coordinates as list
    y_pred : predicted bounding box coordinates as list
    expected shape of arguments: (None,4)
    
    returns: bounding box metric, Intersection-over-Union as list of shape (None,)
    """
    import tensorflow as tf
    from tensorflow.keras import backend as K
    
    with tf.device(GPU):
        # intersection coordinates (x1,y1)
        x1 = tf.maximum(y_true[:,0],y_pred[:,0])
        y1 = tf.maximum(y_true[:,1],y_pred[:,1])
        # intersection coordinates (x2,y2)
        x2 = tf.minimum(y_true[:,2],y_pred[:,2])
        y2 = tf.minimum(y_true[:,3],y_pred[:,3])    
        # compute areas
        inter_area = (x2 - x1) * (y2 - y1)
        union_area = (y_true[:,2] - y_true[:,0]) * (y_true[:,3] - y_true[:,1]) # true bBox
        union_area += (y_pred[:,2] - y_pred[:,0]) * (y_pred[:,3] - y_pred[:,1]) # predicted bBox
        union_area -= inter_area # intersection area
        return (inter_area / (union_area + K.epsilon()))

def IoU(y_true, y_pred):
    """wraps the IOU function into TensorFlow op for eager execution"""
    return tf.py_function(IOU,[y_true, y_pred],tf.float32)
# -------------------------------------------------------------------------------------------------------------------------------
class Pipe():
    """pipeline for ETL-to-Learning-to-Evaluation-&-Reporting"""
    
    def __init__(self,generatorParams,modelFunc,modelName,modelParams={},weights=None):
        """arguments:
        ----------
        generatorParams : dict of kwargs to multiGen(), positional arguments excluded
        modelFunc : method that will return an assembled model
        modelName : representative name for the purpose of record & comparison
        modelParams : dict of arguments to modelFunc, except input_size
        weights : path of pretrained weights file
        """
        
        from lib.extract import trainDF,testDF
        from lib.generator import multiGen
        
        import os
        
        # for future reference
        self.trainDF,self.testDF = trainDF,testDF
        
        # generator implementation
        gParams = dict(train=trainDF,test=testDF,imagePath="ImagePath",
                       className="className",bBox=["x1","y1","x2","y2"],
                       input_size=["width","height"],validation_split=0.1)
        gParams.update(**generatorParams)
        myGen = multiGen(**gParams)
        self.trainset = myGen.subset('training')
        self.validationset = myGen.subset('validation')
        self.testset = myGen.subset('testing')
        self.evalTrain = myGen.subset('evaluation on trainset')
        self.evalTest = myGen.subset('evaluation on testset')
        del myGen
        
        # batch output shape for reference
        self.batch_size = generatorParams.get('batch_size')
        self.target_size = generatorParams.get('target_size')
        self.input_size = tuple((self.batch_size,self.target_size[1],self.target_size[0],3))

        # update model parameters
        self.mParams = dict(input_size=self.input_size)
        self.mParams.update(**modelParams)
        
        # inititate model network
        self.mFunc = modelFunc
        self.model = self.mFunc(**self.mParams)
        self.model.load_weights(weights) if weights!=None else None
        
        # store other arguments
        self.modelName = modelName
        
        # create logs folder
        try:
            os.mkdir('logs')
            os.mkdir('deployables')
        except:
            pass
        
        # record & create model Log
        self.mLog = dict(name=modelName)
        self.mLog.update(**generatorParams)
        
        # fitting status
        self._compiled=False
        self._fitted=False
        
    def compiler(self,optim=tf.keras.optimizers.Adam(),lrScheduleParams={},eStopParams={},
                 trainabilityParams={},gridPoint={},sBest=False):
        """ compile the model & other callbacks
        arguments:
        ----------
        optimParams : dict of arguments to RMSprop, except learning_rate
        lrScheduleParams : dict of arguments to lossGradientLR scheduler, except initial_learning_rate
        eStopParams : dict of arguments to eStop callback instance
        trainabilityParams : dict of arguments to layerUnFreeze callback
        gridPoint : dict of other hyperparameters
        sBest : option to saveBest weights
        """
        
        from lib.customCallbacks import lossGradientLR,LRTensorBoard,eStop
        from lib.customCallbacks import layerUnFreeze,trainableReport,timeLog
        
        import os
        import tensorflow as tf
        from tensorflow.keras import optimizers
        
        # Random consistency seed
        tf.random.set_seed(100)
        
        # losses
        losses = {'names':tf.keras.losses.CategoricalCrossentropy(name='catergoricalCrossEntropy'),
                  'boxes':tf.keras.losses.MeanSquaredError(name='MeanSquaredError')}
        
        loss_weights = {'names':1,'boxes':1}
        loss_weigths = gridPoint.get('lossWeights',loss_weights)

        # metrics
        with tf.device(GPU):
            metrics = {'names':[tf.keras.metrics.CategoricalAccuracy(name='CategoricalAccuracy'),
                                tf.keras.metrics.Precision(name='Precision'),
                                tf.keras.metrics.Recall(name='Recall')],
                       'boxes':IoU}
        
        # compile model
        self.model.compile(optimizer=optim,loss=losses,metrics=metrics,loss_weights=loss_weights)
        # learning rate to be updated before fitting
        
        # create tensorboard logs & callbacks
        logdir = os.path.join(".","logs",self.modelName)
        try:
            os.rmdir(logir)
        except:
            pass
        with tf.device(GPU):
            tensorboard_cb = tf.keras.callbacks.TensorBoard(logdir,histogram_freq=1,embeddings_freq=1)
        
        # initiate learning rate reporter
        lReport = LRTensorBoard(logdir)
        
        # learnin rate decay scheduler
        lrParam = dict(initial_learning_rate=0.555)
        lrParam.update(**lrScheduleParams)
        lr_schedule = lossGradientLR(**lrParam)
        # lr_schedule.lr to be updated before fitting
        
        # save best model weights
        sFile = os.path.join(".","logs",self.modelName,self.modelName+'.h5')
        with tf.device(GPU):
            saveBest = tf.keras.callbacks.ModelCheckpoint(filepath=sFile,monitor='val_loss',
                                                          mode='min',save_best_only=True)
        
        # early stop callback
        eBrake = eStop(**eStopParams)
        
        # layer trainability callback
        trainabilityScheduler = layerUnFreeze(**trainabilityParams)
        
        # model trainability status reporter
        trainabilityReporter = trainableReport(logdir)
        
        # epoch timer call back
        timerCBack = timeLog(logdir)

        # callback collection based on cbOptions
        cbs = [tensorboard_cb,lReport,trainabilityReporter,timerCBack]
        cbs.append(lr_schedule) if lrScheduleParams!={} else None
        cbs.append(eBrake) if eStopParams!={} else None
        cbs.append(trainabilityScheduler) if trainabilityParams!={} else None
        cbs.append(saveBest) if sBest else None

        # update fitParams
        self.fParams = dict(callbacks=cbs)
        
        # record & update model Log
        self.mLog.update(dict(optimizer=optim._name))
        self.mLog.update(**lrScheduleParams)
        self.mLog.update(**eStopParams)
        self.mLog.update(**trainabilityParams)
        self.mLog.update(**gridPoint)
        
        # update comilation status
        self._compiled = True
        
    def fit(self,gridPoint,fitParams):
        """ learner method
        arguments:
        ----------
        gridPoint : dict of hyperparameters of the model, MUST include learning_rate
        fitParams : dict of arguments to model.fit(), except dataset selection
        """
        
        assert self._compiled, "attempt to fit uncompiled model"
        
        import tensorflow as tf
        from tensorflow.keras import backend as K

        from lib.customCallbacks import lossGradientLR
        
        # Random consistency seed
        tf.random.set_seed(100)
        
        # update optimiser learning rate
        lr = gridPoint.get('learning_rate',0.555)
        K.set_value(self.model.optimizer.learning_rate, lr)
        
        # update initial_epoch
        init_epoch = fitParams.get('initial_epoch',0)
        self.trainset.epoch = init_epoch
        self.validationset.epoch = init_epoch
        self.testset.epoch = init_epoch
        
        # update learning rate scheduler initial value
        cbs = self.fParams.get('callbacks')
        for cb in cbs:
            if isinstance(cb,lossGradientLR):  # if scheduler was opted
                cb.lr = [lr]
                cb.lastAdjust = 0
        
        # update fitParams
        self.fParams.update(**fitParams)
        
        # model fitting
        with tf.device(GPU):
            self.logger = self.model.fit(self.trainset,validation_data=self.validationset,**self.fParams)
            
        # record & update model Log
        self.mLog.update(**gridPoint)
        self.mLog.update(**fitParams)
        self.mLog.update(logs=self.logger.history)
            
        self._fitted=True                 
        
    def visualiseFit(self):
        """display the model training history metrics"""
        
        assert self._fitted, "unfit model"
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme()
        
        df = pd.DataFrame(self.logger.history)
        
        plt.figure(figsize=(16,8))        

        plt.subplot(3,4,1)
        ax = sns.lineplot(data=df,y='learning_rate',x=df.index)
        ax.set_yscale('log')
        plt.title('learning_rate')
        ax.legend(['scheduled'])
        ax.set(xlabel=None,ylabel=None);
        
        plt.subplot(3,4,5)
        ax = sns.lineplot(data=df,y='lapTime',x=df.index)
        plt.title('resource consumption')
        ax.legend(['lapTime(s)'])
        ax.set(xlabel=None,ylabel=None);
        
        plt.subplot(3,4,9)
        total = len(self.model.layers)
        ax = sns.lineplot(data=df,y='Trainable',x=df.index) # trainable layer count
        ax = sns.lineplot(y=[total]*df.shape[0],x=df.index) # full layer count
        ax.legend(['trainable','total'])
        plt.title('Count of Trainable layers')
        ax.set(xlabel='epoch',ylabel=None);

        plt.subplot(2,4,2)
        ax = sns.lineplot(data=df,y='boxes_IoU',x=df.index)
        ax = sns.lineplot(data=df,y='val_boxes_IoU',x=df.index)
        ax.legend(['training','validation'])
        plt.title('Bounding Box IOU')
        ax.set(xlabel=None,ylabel=None);

        plt.subplot(2,4,6)
        ax = sns.lineplot(data=df,y='boxes_loss',x=df.index)
        ax = sns.lineplot(data=df,y='val_boxes_loss',x=df.index)
        ax.legend(['training','validation'])
        plt.title('Bounding Box loss')
        ax.set(xlabel='epoch',ylabel=None);

        plt.subplot(2,4,3)
        ax = sns.lineplot(data=df,y='names_CategoricalAccuracy',x=df.index)
        ax = sns.lineplot(data=df,y='val_names_CategoricalAccuracy',x=df.index)
        ax.legend(['training','validation'])
        plt.title('categorical accuracy')
        ax.set(xlabel=None,ylabel=None);

        plt.subplot(2,4,7)
        ax = sns.lineplot(data=df,y='names_loss',x=df.index)
        ax = sns.lineplot(data=df,y='val_names_loss',x=df.index)
        ax.legend(['training','validation'])
        plt.title('categorical crossentropy')
        ax.set(xlabel='epoch',ylabel=None);

        plt.subplot(2,4,4)
        ax = sns.lineplot(data=df,y='names_Precision',x=df.index)
        ax = sns.lineplot(data=df,y='val_names_Precision',x=df.index)
        ax.legend(['training','validation'])
        plt.title('average precision')
        ax.set(xlabel=None,ylabel=None);

        plt.subplot(2,4,8)
        ax = sns.lineplot(data=df,y='names_Recall',x=df.index)
        ax = sns.lineplot(data=df,y='val_names_Recall',x=df.index)
        ax.legend(['training','validation'])
        plt.title('average recall')
        ax.set(xlabel='epoch',ylabel=None);
        
    def evaluate(self):
        """evaluate on train & test data"""
        
        assert self._fitted, "unfit model"
        """
        print("\n----------")
        print("EVALUATION")
        print("----------\n")"""
        
        import tensorflow as tf
        
        # evaluate on training data
        with tf.device(GPU):
            scores = self.model.evaluate(self.evalTrain,verbose=0)
        evalCols = ['loss', 'names_loss', 'boxes_loss', 'names_CategoricalAccuracy',
                    'names_Precision', 'names_Recall', 'boxes_IoU']
        self.mLog.update(dict(zip(evalCols,scores)))
        # evaluate on testing data
        with tf.device(GPU):
            scores = self.model.evaluate(self.evalTest,verbose=0)
        evalCols = ['val_'+ec for ec in evalCols]
        self.mLog.update(dict(zip(evalCols,scores)))
        
    def report(self):
        """report model performance on train & test data"""
        
        assert self._fitted, "unfit model"
        
        import tensorflow as tf
        import numpy as np
        import pandas as pd
        from sklearn import metrics
        pd.set_option('display.max_columns', 50)
        
        # display training history
        self.visualiseFit() 
        
        # display Logs
        dLog = pd.DataFrame(data=np.array(list(self.mLog.values()),
                                          dtype=object).reshape(1,-1),
                            columns=list(self.mLog.keys()),dtype=object)
        display(dLog)
        
        """print("\n----------")
        print("PREDICTION")
        print("----------\n")"""
        
        # predict on training data       
        with tf.device(GPU):
            [predTrainClass,predTrainBox] = self.model.predict(self.evalTrain,verbose=0)
        tr_len = predTrainBox.shape[0]
        [trainClass,trainBox] = [self.evalTrain.cName[:tr_len],self.evalTrain.bBox[:tr_len]]
        predTrainBox = self.evalTrain.unSize(predTrainBox,np.arange(tr_len))
        trainImages = self.evalTrain.imPath[:tr_len]
        
        # predict on testing data
        with tf.device(GPU):
            [predTestClass,predTestBox] = self.model.predict(self.evalTest,verbose=0)
        tt_len = predTestBox.shape[0]
        [testClass,testBox] = [self.evalTest.cName[:tt_len],self.evalTest.bBox[:tt_len]]
        predTestBox = self.evalTest.unSize(predTestBox,np.arange(tt_len))
        testImages = self.evalTest.imPath[:tt_len]
        
        # obtain confusion matrix
        confTrain = metrics.multilabel_confusion_matrix(trainClass,(predTrainClass>=0.5)*1)
        confTest = metrics.multilabel_confusion_matrix(testClass,(predTestClass>=0.5)*1)
        
        enc = self.trainset.enc
        # visualise confusion metrics
        confPlot1(enc.classes_,confTrain,confTest)
        confPlot2(enc.classes_,confTrain,confTest) # testing 2 types of visualisation
        
        # visualise few images with predictions on training set
        print("samples from TRAINING SET")
        sampleResult(trainImages,trainBox,predTrainBox,
                     enc.inverse_transform(trainClass),enc.inverse_transform(predTrainClass))
        print("samples from TESTING SET")
        sampleResult(testImages,testBox,predTestBox,
                     enc.inverse_transform(testClass),enc.inverse_transform(predTestClass))
        
        """self.resLog = mReport(self.modelName,
                              self.evalTrain.imPath[:tr_len],[trainClass,predTrainClass,trainBox,predTrainBox],
                              self.evalTest.imPath[:tt_len],[testClass,predTestClass,testBox,predTestBox],
                              self.trainset.enc,verbose=1)"""

    def htuneSave(self):
        """saving only the logs for hypertuning purpose"""
        assert self._fitted, "unfit model"
        import pickle
        import os
        path = os.path.join('.',"deployables")
        fname = os.path.join(path,"%s_attrib.gl"%self.modelName) # file name
        attribNames = ['labelEncoder','modelFunc','inputShape','evalResults']
        attributes = [self.trainset.enc,self.mFunc,self.model.input_shape,self.mLog]
        with open(fname, 'wb') as fh:
            pickle.dump(dict(zip(attribNames,attributes)), fh) # pickling
        
    def save(self):
        """pickle each critical components to create a deployable model"""
        
        assert self._fitted, "unfit model"
        
        import pickle
        import os
        
        path = os.path.join('.',"deployables")
        try:
            os.listdir(path) # check availability
        except:
            os.mkdir(path) # create folder
        path = os.path.join(path,self.modelName)
        try:
            os.rmdir(path) # delete duplicate folder
            os.mkdir(path) # create fresh folder
        except:
            try:
                os.mkdir(path) # create fresh folder
            except:
                pass 
        
        fname = os.path.join(path,"%s_attrib.gl"%self.modelName) # file name
        attribNames = ['labelEncoder','modelFunc','inputShape','evalResults']
        attributes = [self.trainset.enc,self.mFunc,self.model.input_shape,self.mLog]
        with open(fname, 'wb') as fh:
            pickle.dump(dict(zip(attribNames,attributes)), fh) # pickling
        fname = os.path.join(path,"%s_model.h5"%self.modelName) 
        self.model.save(fname) # h5 model save
        fname = os.path.join(path,"%s_weights.h5"%self.modelName) 
        self.model.save_weights(fname) # model weights
        
        tfpath = os.path.join(path,self.modelName,"tfsave")
        try:
            os.mkdir(path) # create tf save folder
        except:
            pass
        self.model.save(tfpath) # tf save
# -------------------------------------------------------------------------------------------------------------------------------
# evaluation & logging
lCols = ['modelName','loss','names_loss','boxes_loss','names_accuracy','names_precision','names_recall',
         'boxes_IoU','val_loss','val_names_loss','val_boxes_loss','val_names_accuracy','val_names_precision',
         'val_names_recall','val_boxes_IoU']

def mReport(mName,trainImages,outSetA,testImages,outSetB,enc,verbose=1):
    """arguments:
    ----------
    mName : model names as string
    trainImages : list of training image paths
    outSetA : outputs for training dataset [trueY,predY,trueYbox,predYbox]
    testImages : list of testing image paths
    outSetB : outputs for testing dataset [testY,predTestY,testYbox,predTestYbox]
    enc: fitted LabelBinarizer() object
    
    note: all box coordinates to be rescaled to orignal aspect ratios & sizes
    
    returns: None
    
    functionality:
    --------------
    updates resLog
    displays classification report & confusion stats
    displays sample results for training & testing datasets
    """
    import pandas as pd
    import numpy as np
    from sklearn import metrics
    
    trueY,predY,trueYbox,predYbox = outSetA
    testY,predTestY,testYbox,predTestYbox = outSetB
    
    mLog = np.array([mName],dtype=object)
    mLog = evalLog(mLog,*outSetA)
    mLog = evalLog(mLog,*outSetB)

    # append to results log
    resLog = pd.DataFrame(mLog.reshape(1,-1),columns=lCols)
    display(resLog)
    resLog = dict(zip(lCols,mLog))
    
    if not verbose:
        return resLog
    
    # obtain confusion matrix
    confTrain = metrics.multilabel_confusion_matrix(trueY,(predY>=0.5)*1)
    confTest = metrics.multilabel_confusion_matrix(testY,(predTestY>=0.5)*1)
    
    # visualise confusion metrics
    confPlot1(enc.classes_,confTrain,confTest)
    confPlot2(enc.classes_,confTrain,confTest) # testing 2 types of visualisation
    
    # visualise few images with predictions on training set
    print("samples from TRAINING SET")
    sampleResult(trainImages,trueYbox,predYbox,enc.inverse_transform(trueY),enc.inverse_transform(predY))
    print("samples from TESTING SET")
    sampleResult(testImages,testYbox,predTestYbox,enc.inverse_transform(testY),enc.inverse_transform(predTestY))
    
    return resLog

def interBox(y_true,y_pred):
    """arguments:
    ----------
    y_true : true bounding box coordinates as list
    y_pred : predicted bounding box coordinates as list
    expected shape of arguments: (None,4)
    
    returns: coordinates of intersection of true & predicted bounding boxes (None,4)
    """
    import numpy as np
    # intersection coordinates (x1,y1)
    x1 = np.maximum(y_true[:,0],y_pred[:,0])
    y1 = np.maximum(y_true[:,1],y_pred[:,1])
    # intersection coordinates (x2,y2)
    x2 = np.minimum(y_true[:,2],y_pred[:,2])
    y2 = np.minimum(y_true[:,3],y_pred[:,3]) 
    
    return np.array((x1,y1,x2,y2)).transpose()

def evalLog(mLog,trueY,predY,trueBox,predBox):    
    """arguments:
    ----------
    mLog : array of current model metrics
    trueClass : binarized true labels (None,196)
    predClass : predicted logits (None,196)
    trueBox : true bounding box coordinates (None,4)
    predBox : predicted bounding box coordinates (None,4)
    
    returns : updated array of current model metrics
    """
    import tensorflow as tf
    import numpy as np
    
    with tf.device(GPU):
        
        # log of metrics & losses for training dataset
        names_loss = tf.keras.losses.CategoricalCrossentropy()(trueY,predY).numpy()
        boxes_loss = tf.keras.losses.MeanSquaredError()(trueBox,predBox).numpy()
        loss_sum = names_loss+boxes_loss

        mLog = np.append(mLog,loss_sum)
        mLog = np.append(mLog,names_loss)
        mLog = np.append(mLog,boxes_loss)
        mLog = np.append(mLog,tf.keras.metrics.CategoricalAccuracy()(trueY,predY).numpy())
        mLog = np.append(mLog,tf.keras.metrics.Precision()(trueY,predY).numpy())
        mLog = np.append(mLog,tf.keras.metrics.Recall()(trueY,predY).numpy())
        mLog = np.append(mLog,IoU(trueBox.astype('float32'),predBox.astype('float32')).numpy().mean())
    
    return mLog

def confPlot1(labels,confTrain,confTest):
    """arguments:
    ----------
    labels : list of classes
    confTrain : multi-label confusion matrix for prediction on training set
    confTest : multi-label confusion matrix for prediction on test set
    
    functionality:
    --------------
    displays subplots pf confusion metrics
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np    
    sns.set_theme()
    
    xls = np.arange(len(labels))
    plt.figure(figsize=(16,5))
    plt.suptitle('TRAINING DATASET                                                                                               TESTING DATASET')
    
    p1 = plt.subplot(2,4,1)
    plt.fill_between(x=xls,y1=confTrain[:,0,0],color='#FFA15A')
    p1.xaxis.set_ticks([])
    plt.title("TRUE NEGATIVES")
    
    p2 = plt.subplot(2,4,2)#,sharey=p1)
    plt.fill_between(x=xls,y1=confTrain[:,0,1],color='#19D3F3')
    p2.xaxis.set_ticks([])
    plt.title("FALSE POSITIVES")
    
    p5 = plt.subplot(2,4,5)
    plt.fill_between(x=xls,y1=confTrain[:,1,0],color='#B82E2E')
    p5.xaxis.set_ticks([])
    plt.title("FALSE NEGATIVES")
    
    p6 = plt.subplot(2,4,6)#,sharey=p5)
    plt.fill_between(x=xls,y1=confTrain[:,1,1],color='#109618')
    p6.xaxis.set_ticks([])
    plt.title("TRUE POSITIVES")
    
    p3 = plt.subplot(2,4,3)#,sharey=p1)
    plt.fill_between(x=xls,y1=confTest[:,0,0],color='#FFA15A')
    p3.xaxis.set_ticks([])
    plt.title("TRUE NEGATIVES")
    
    p4 = plt.subplot(2,4,4)#,sharey=p1)
    plt.fill_between(x=xls,y1=confTest[:,0,1],color='#19D3F3')
    p4.xaxis.set_ticks([])
    plt.title("FALSE POSITIVES")
    
    p7 = plt.subplot(2,4,7)#,sharey=p5)
    plt.fill_between(x=xls,y1=confTest[:,1,0],color='#B82E2E')
    p7.xaxis.set_ticks([])
    plt.title("FALSE NEGATIVES")
    
    p8 = plt.subplot(2,4,8)#,sharey=p5)
    plt.fill_between(x=xls,y1=confTest[:,1,1],color='#109618')
    p8.xaxis.set_ticks([])
    plt.title("TRUE POSITIVES")
    
    plt.show()
    
def confPlot2(labels,confTrain,confTest):
    """arguments:
    ----------
    labels : list of classes
    confTrain : multi-label confusion matrix for prediction on training set
    confTest : multi-label confusion matrix for prediction on test set
    
    functionality:
    --------------
    displays overlayed confusion metrics in log scale
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np    
    sns.set_theme()
    
    plt.figure(figsize=(16,5))
    xls = np.arange(len(labels))
    
    p1=plt.subplot(1,2,1)
    plt.title("TRAINING DATASET")
    plt.fill_between(x=xls,y1=confTrain[:,0,0],color='#FFA15A')
    plt.fill_between(x=xls,y1=confTrain[:,0,1],color='#19D3F3')
    plt.fill_between(x=xls,y1=confTrain[:,1,0],color='#B82E2E')
    plt.fill_between(x=xls,y1=confTrain[:,1,1],color='#109618')
    p1.xaxis.set_ticks([])
    plt.yscale("log") 
    
    p2=plt.subplot(1,2,2,sharey=p1)
    plt.title("TESTING DATASET")
    plt.fill_between(x=xls,y1=confTest[:,0,0],color='#FFA15A')
    plt.fill_between(x=xls,y1=confTest[:,0,1],color='#19D3F3')
    plt.fill_between(x=xls,y1=confTest[:,1,0],color='#B82E2E')
    plt.fill_between(x=xls,y1=confTest[:,1,1],color='#109618')
    p2.xaxis.set_ticks([])
    plt.yscale("log")
    
    plt.show()
    
def sampleResult(imPath,trueBox,predBox,trueClass,predClass):
    """ display 3 sample images with true & predicted targets
    arguments:
    ----------
    imPath : paths of images
    trueBox : true coordinates of bounding box
    predBox : predicted coordinates of bounding box
    trueClass : list of true class names (strings)
    predClass : list of predicted class names (strings)
    """
    import tensorflow as tf
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns  
    import cv2
    sns.set_theme()
    
    tkpi = tf.keras.preprocessing.image
    indices = np.random.choice(range(len(imPath)),3,replace=False) # displays 3 sample images
    plt.figure(figsize=(16,5))
    for i,index in enumerate(indices):
        ax = plt.subplot(1,len(indices),i+1)
        img = tkpi.img_to_array(tkpi.load_img(imPath[index]))
        tBox = trueBox[index]    
        pBox = predBox[index]
        iBox = interBox(tBox.reshape(-1,4),pBox.reshape(-1,4))[0]
        cv2.rectangle(img,tBox[:2],tBox[2:],(0,255,0),2)
        cv2.rectangle(img,pBox[:2],pBox[2:],(255,0,255),2)
        cv2.rectangle(img,iBox[:2],iBox[2:],(0,0,255),4)
        img = tkpi.array_to_img(img)
        plt.imshow(img)
        plt.title("True Class: %s"%trueClass[index],fontdict=dict(fontsize=10))
        plt.xlabel("Prediction: %s"%predClass[index])
        plt.ylabel("IOU: %.4f"%IOU(tBox.reshape(-1,4)*1.0,pBox.reshape(-1,4)*1.0))
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
    plt.show()
