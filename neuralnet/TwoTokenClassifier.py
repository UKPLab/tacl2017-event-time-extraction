import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Merge
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.utils import np_utils
from BaseModel import BaseModel
import cPickle as pkl
import time
import os

class TwoTokenClassifier(BaseModel):
    def __init__(self, name, additionalFeatures=False):
        BaseModel.__init__(self, name)
        
        f = open('pkl/'+name+'/data.pkl', 'rb')
        self.trainSet  = pkl.load(f)
        self.devSet = pkl.load(f)
        self.testSet = pkl.load(f)
        f.close()
        
        f = open('pkl/embeddings.pkl', 'rb')
        self.embeddings = pkl.load(f)
        f.close()

        self.modelOutputPath = 'models/'+name  
        if not os.path.exists(self.modelOutputPath):
            os.makedirs(self.modelOutputPath)
       
        self.np_epoch = 20
        
        self.featureSet = ['event', 'time', 'sentence', 'positions_e', 'positions_t']
        
        self.space = {
             'update_word_embeddings': [False, True],             
             'nb_filter': range(5,500+1,5),
             'filter_length': [1,2,3,4,5],
             'batch_size': [32,64,128],
             'hidden_dims': range(5,300+1,5),
             'position_dims': range(5,30+1),
             'additional_features_dims': range(5,100+1),
             'activation_cnn': ['relu', 'tanh'],
             'activation_dense':  ['relu', 'tanh'],
             'dropout1': [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75 ],
             'dropout2': [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75 ],
             'optimizer': ['adagrad', 'adadelta','adam','adamax','rmsprop', 'nadam'],
             'use_positions': [False, True],
             'additional_features': self.powerset(['tense', 'aspect', 'eventClass', 'sentence_len', 'type'])
            }
            
        if not additionalFeatures:
        	  self.space['additional_features'] = ['']
        
    def run_model(self, params, best_model_acc):
        
        
        modelOutputPath = self.modelOutputPath
        trainSet = self.trainSet
        devSet = self.devSet
        testSet = self.testSet
        embeddings = np.copy(self.embeddings)
        nb_epoch = self.np_epoch
        featureSet = list(self.featureSet) + list(params['additional_features'])
        
        start_time = time.time()  
    
        batch_size = params['batch_size']
        nb_filter = params['nb_filter']
        filter_length = params['filter_length']
        hidden_dims = params['hidden_dims']
        position_dims = params['position_dims']
        additional_features_dims = params['additional_features_dims']
        use_positions = params['use_positions']
        
       
    
        n_out = max(trainSet['labels'])+1
        train_y_cat = np_utils.to_categorical(trainSet['labels'], n_out)
     
        eventModel = Sequential()
        eventModel.add(Embedding(embeddings.shape[0], embeddings.shape[1], input_length=1, weights=[embeddings], trainable=params['update_word_embeddings']))
        eventModel.add(Flatten())
    
        timeModel = Sequential()
        timeModel.add(Embedding(embeddings.shape[0], embeddings.shape[1], input_length=1, weights=[embeddings], trainable=params['update_word_embeddings']))
        timeModel.add(Flatten())
    
        # :: Extra features ::
        extraFeaturesModels = []
        
        for add_feature in params['additional_features']:
            extraFeatureModel = Sequential()
            extraFeatureModel.add(Embedding(np.max(trainSet[add_feature])+1, additional_features_dims, input_length=trainSet[add_feature].shape[1]))
            extraFeatureModel.add(Flatten())
            
            extraFeaturesModels.append(extraFeatureModel)
         
        """   
        tenseModel = Sequential()
        tenseModel.add(Embedding(np.max(trainSet['tense'])+1, additional_features_dims, input_length=trainSet['tense'].shape[1]))
        tenseModel.add(Flatten())
        
        aspectModel = Sequential()
        aspectModel.add(Embedding(np.max(trainSet['aspect'])+1, additional_features_dims, input_length=trainSet['aspect'].shape[1]))
        aspectModel.add(Flatten())
        
        eventClassModel = Sequential()
        eventClassModel.add(Embedding(np.max(trainSet['eventClass'])+1, additional_features_dims, input_length=trainSet['eventClass'].shape[1]))
        eventClassModel.add(Flatten())
    
        typeModel = Sequential()
        typeModel.add(Embedding(np.max(trainSet['type'])+1, additional_features_dims, input_length=trainSet['type'].shape[1]))
        typeModel.add(Flatten())
        """
        
        # :: Sentence CNN ::
        
        wordModel = Sequential()
        wordModel.add(Embedding(embeddings.shape[0], embeddings.shape[1], input_length=trainSet['sentence'].shape[1], weights=[embeddings], trainable=params['update_word_embeddings']))
       
        max_position = max(np.max(trainSet['positions_e']), np.max(trainSet['positions_t']))+1
    
        if use_positions:
            distanceModel_e = Sequential()
            distanceModel_e.add(Embedding(max_position, position_dims, input_length=trainSet['positions_e'].shape[1]))
        
            distanceModel_t = Sequential()
            distanceModel_t.add(Embedding(max_position, position_dims, input_length=trainSet['positions_t'].shape[1]))
        
            convModel = Sequential()
            convModel.add(Merge([wordModel, distanceModel_e, distanceModel_t], mode='concat'))           
        else:
            convModel = wordModel
            featureSet.remove('positions_e')
            featureSet.remove('positions_t')
        
        # we add a Convolution1D, which will learn nb_filter
        # word group filters of size filter_length:
        convModel.add(Convolution1D(nb_filter=nb_filter,
                                filter_length=filter_length,
                                border_mode='same',
                                activation=params['activation_cnn'],
                                subsample_length=1))
        # we use standard max pooling
        convModel.add(MaxPooling1D(pool_length=convModel.output_shape[1]))
        convModel.add(Flatten())
    
        
        
        # :: Combine all models ::
        
        model = Sequential()
        model.add(Merge([eventModel, timeModel, convModel]+extraFeaturesModels, mode='concat'))
        
        model.add(Dropout(params['dropout1']))
        model.add(Dense(hidden_dims,  activation=params['activation_dense']))
        model.add(Dropout(params['dropout2']))
        # We project onto a single unit output layer, and squash it with a sigmoid:
        model.add(Dense(n_out, activation='softmax'))
        
        
        model.compile(loss='categorical_crossentropy',optimizer=params['optimizer'])
        
     
    
        print params
        print featureSet
      
        
        max_acc = 0
        for epoch in xrange(nb_epoch):   
            model.fit([trainSet[ft] for ft in featureSet], train_y_cat, batch_size=batch_size, verbose=False, nb_epoch=1)
            
            acc_test, prec_test, rec_test, f1_test = self.getAccuracy(model, [testSet[ft] for ft in featureSet], testSet['labels'])
            max_acc = max(acc_test, max_acc)
            
            if acc_test > best_model_acc:
                acc_dev, prec_dev, rec_dev, f1_dev = self.getAccuracy(model, [devSet[ft] for ft in featureSet], devSet['labels'])
               
               
                print "Acc: %.4f | F1: %.4f on dev" % (acc_dev, f1_dev) 
                print "Acc: %.4f | F1: %.4f on test\n" % (acc_test, f1_test)
                #Save the model
                modelOutputPathName = modelOutputPath+'/%.4f_%.4f_model.h5' % (acc_test, acc_dev)
                model.save(modelOutputPathName)                
                self.save_params(params, featureSet, modelOutputPathName)
                
                best_model_acc = acc_test
        
        
        print '%.2f sec for training' % (time.time() - start_time)
        return max_acc, best_model_acc
