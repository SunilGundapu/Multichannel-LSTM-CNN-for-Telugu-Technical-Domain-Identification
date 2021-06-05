#!/usr/bin/env python
# coding: utf-8

# In[8]:


import csv, sys, string
import pandas as pd
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
import numpy as np
import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF
from keras.preprocessing import text, sequence
import warnings
import pickle
warnings.simplefilter("ignore")


from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints



class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        #self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim
        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))
        if self.bias:
            eij += self.b
        eij = K.tanh(eij)
        a = K.exp(eij)
        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())
        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
    #print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        #return input_shape[0], input_shape[-1]
        return input_shape[0],  self.features_dim



def getData(file):
    tsv_file = open(file)
    df = pd.read_csv(tsv_file, delimiter="\t")
    return df

def labelEncoder(labels):
    encoder = preprocessing.LabelEncoder()
    tempLabels = encoder.fit_transform(labels)
    tempLabels = [to_categorical(i, num_classes=6) for i in tempLabels]
    tempLabels = np.asarray(tempLabels)
    return tempLabels

def train_model(classifier, feature_vector_train, label, feature_vector_valid, epoch=False, is_neural_net=False):
    if is_neural_net:
        classifier.fit(feature_vector_train, label, epochs=epoch)
        # predict the labels on validation dataset
        predictions = classifier.predict(feature_vector_valid)
        predictions1 = predictions.argmax(axis=-1)
        validLabels1 = validLabels.argmax(axis=-1)
        #print(predictions, predictions1)
        #print(validLabels, validLabels1)
        acc = metrics.accuracy_score(predictions1, validLabels1)
        f1Score = metrics.f1_score(predictions1, validLabels1, average='macro')
    else:
        # fit the training dataset on the classifier
        classifier.fit(feature_vector_train, label)
        # predict the labels on validation dataset
        predictions = classifier.predict(feature_vector_valid)
        acc = metrics.accuracy_score(predictions, validDF['label'])
        f1Score = metrics.f1_score(predictions, validDF['label'], average='macro')
    
    return acc, f1Score


trainFilename = "sub-task-1h/sub-task-1h-train-te.tsv"
validFilename = "sub-task-1h/sub-task-1h-dev-te.tsv"
trainDF = getData(trainFilename)
validDF = getData(validFilename)
print("Train Data Shape: ",trainDF.shape)
print("Validation Data Shape: ",validDF.shape)
totalTextData = pd.concat([trainDF['text'], validDF['text']])

transformLabels = preprocessing.LabelEncoder()
transformLabels.fit(['phy','cse','other','com_tech','bioche','mgmt'])
trainDF['numericalLabels'] = transformLabels.transform(trainDF['label'])
validDF['numericalLabels'] = transformLabels.transform(validDF['label'])

encoder = preprocessing.LabelEncoder()
trainLabels = labelEncoder(trainDF['label'])
validLabels = labelEncoder(validDF['label'])
print(trainLabels.shape)
print(validLabels.shape)

# create a tokenizer 
token = text.Tokenizer()
token.fit_on_texts(totalTextData)
word_index = token.word_index
maxSentenceLength = 150

# convert text to sequence of tokens and pad them to ensure equal length vectors 
train_seq_x = sequence.pad_sequences(token.texts_to_sequences(trainDF['text']), maxlen=maxSentenceLength)
valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(validDF['text']), maxlen=maxSentenceLength)

#Load pickle
with open('wordEmbeddingsMatrix.pickle', 'rb') as handle:
    embedding_matrix = pickle.load(handle)

def create_rnn_lstm(input_size = maxSentenceLength):    
    # Add an Input Layer
    input_layer = layers.Input((input_size, ))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
    #embedding_layer = layers.SpatialDropout1D(0.2)(embedding_layer)

    # Add the LSTM Layer
    lstm_layer = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(embedding_layer)
    
    #Self Attention
    attention_layer = Attention(input_size)(lstm_layer)
    
    # Add the output Layers
    output_layer1 = layers.Dense(122, activation="relu")(attention_layer)
    #output_layer1 = layers.Dropout(0.2)(output_layer1)
    output_layer2 = layers.Dense(6, activation="softmax")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='categorical_crossentropy')
    
    return model

classifier = create_rnn_lstm(maxSentenceLength)
accuracy = train_model(classifier, train_seq_x, trainLabels, valid_seq_x,1, is_neural_net=True)
print("RNN-LSTM, Word Embeddings",  accuracy)

