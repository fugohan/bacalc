import keras as k
import tensorflow as tf
from  functions.datasetLoader import datasetLoader, minmaxindex
from sklearn.preprocessing import MinMaxScaler
# import numpy as np
from functions.windowGen import windowGen
from sklearn.model_selection import train_test_split
from keras  import callbacks
from keras import optimizers
from keras.callbacks import EarlyStopping
#
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, SimpleRNN, GRU
#
import datetime as dt 
############################################################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras as k
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
#####################
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import GRU, Dense
from keras.layers import LSTM, Dropout
from keras  import callbacks
from keras import optimizers
from keras.callbacks import EarlyStopping
#####################
from functions.windowGen import windowGen
from  functions.datasetLoader import datasetLoader, minmaxindex
from functions.plot_style import plot_style
##################
from IPython.display import Image
from keras.utils import plot_model
###################
# from sklearn.externals import joblib
import joblib
############################################################################
def testmodel1(dataset="litecoin",sequence_length=1,
lookback=1,test_size=0.3, epochs=100000, batch_size=128,
neurons=128, recurrent_dropout=0.08, dropout=0.01, dense=1, loss= "mse", optimiser = "adam", patience=100):
  dataset_data  = datasetLoader("dataset/"+dataset+".csv")
  data = dataset_data.values
  scaler = MinMaxScaler()
  data_scaled  = scaler.fit_transform(data.reshape(-1,1))
  #
  # if sequence_length is not 1: 
  X, y = windowGen(data_scaled, sequence_length,lookback)
  X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size, shuffle=False)
  input_shape = (sequence_length, 1)
  s =dt.datetime.now()
  model = Sequential()
  model.add(LSTM(neurons, input_shape=input_shape, return_sequences=False, recurrent_dropout=recurrent_dropout, dropout=dropout))
  model.add(Dense(dense))
  model.compile(loss=loss, optimizer=optimiser)
  monitor = EarlyStopping(monitor='loss', patience=patience, verbose=0, mode='auto', restore_best_weights=True)
  history = model.fit(X_train,y_train,validation_data=(X_test,y_test), callbacks=[monitor],verbose=0,epochs=epochs,batch_size=batch_size)#added batchsize
  e =dt.datetime.now()
  time = e-s
  return scaler, model, history , X_train, X_test, y_train, y_test, time

def testmodel2(dataset="litecoin",sequence_length=1, lookback=1,test_size=0.3, 
epochs=100000, batch_size=128, neurons=128, recurrent_dropout=0.08, dropout=0.01, 
dense=1, loss= "mse", optimiser = "adam", patience=100):
  dataset_data  = datasetLoader("dataset/"+dataset+".csv")
  data = dataset_data.values
  scaler = MinMaxScaler()
  data_scaled  = scaler.fit_transform(data.reshape(-1,1))
  #
  # if sequence_length is not 1: 
  X, y = windowGen(data_scaled, sequence_length,lookback)
  X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size, shuffle=False)
  input_shape = (sequence_length, 1)
  s =dt.datetime.now()
  model = Sequential()
  model.add(SimpleRNN(neurons, input_shape=input_shape, return_sequences=False, recurrent_dropout=recurrent_dropout, dropout=dropout))
  model.add(Dense(dense))
  model.compile(loss=loss, optimizer=optimiser)
  monitor = EarlyStopping(monitor='loss', patience=patience, verbose=0, mode='auto', restore_best_weights=True)
  history = model.fit(X_train,y_train,validation_data=(X_test,y_test), callbacks=[monitor],verbose=2,epochs=epochs,batch_size=batch_size)#added batchsize
  e =dt.datetime.now()
  time = e-s
  return scaler, model, history , X_train, X_test, y_train, y_test, time

def testmodel3(dataset="litecoin",sequence_length=1, lookback=1,test_size=0.3, 
epochs=100000, batch_size=128, neurons=128, recurrent_dropout=0.08, dropout=0.01, 
dense=1, loss= "mse", optimiser = "adam", patience=100):
  dataset_data  = datasetLoader("dataset/"+dataset+".csv")
  data = dataset_data.values
  scaler = MinMaxScaler()
  data_scaled  = scaler.fit_transform(data.reshape(-1,1))
  #
  # if sequence_length is not 1: 
  X, y = windowGen(data_scaled, sequence_length,lookback)
  X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size, shuffle=False)
  input_shape = (sequence_length, 1)
  s =dt.datetime.now()
  model = Sequential()
  model.add(GRU(neurons, input_shape=input_shape, return_sequences=False, recurrent_dropout=recurrent_dropout, dropout=dropout))
  model.add(Dense(dense))
  model.compile(loss=loss, optimizer=optimiser)
  monitor = EarlyStopping(monitor='loss', patience=patience, verbose=0, mode='auto', restore_best_weights=True)
  history = model.fit(X_train,y_train,validation_data=(X_test,y_test), callbacks=[monitor],verbose=2,epochs=epochs,batch_size=batch_size)#added batchsize
  e =dt.datetime.now()
  time = e-s
  return scaler, model, history , X_train, X_test, y_train, y_test, time
def testmodel4(dataset="litecoin",sequence_length=1, lookback=1,test_size=0.3, 
epochs=100000, batch_size=128, neurons=128, recurrent_dropout=0.08, dropout=0.01, 
dense=1, loss= "mse", optimiser = "adam", patience=100):
  dataset_data  = datasetLoader("dataset/"+dataset+".csv")
  data = dataset_data.values
  scaler = MinMaxScaler()
  data_scaled  = scaler.fit_transform(data.reshape(-1,1))
  #
  # if sequence_length is not 1: 
  X, y = windowGen(data_scaled, sequence_length,lookback)
  X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size, shuffle=False)
  input_shape = (sequence_length, 1)
  s =dt.datetime.now()
  model = Sequential()
  model.add(GRU(neurons, input_shape=input_shape, return_sequences=True, recurrent_dropout=recurrent_dropout, dropout=dropout))
  model.add(GRU(neurons*2, input_shape=input_shape, return_sequences=True, recurrent_dropout=recurrent_dropout, dropout=dropout))
  model.add(GRU(neurons*3, return_sequences=False, recurrent_dropout=recurrent_dropout, dropout=dropout))
  model.add(Dense(dense))
  model.compile(loss=loss, optimizer=optimiser)
  monitor = EarlyStopping(monitor='loss', patience=patience, verbose=0, mode='auto', restore_best_weights=True)
  history = model.fit(X_train,y_train,validation_data=(X_test,y_test), callbacks=[monitor],verbose=2,epochs=epochs,batch_size=batch_size)#added batchsize
  e =dt.datetime.now()
  time = e-s
  return scaler, model, history , X_train, X_test, y_train, y_test, time

def testmodel5(dataset="litecoin",sequence_length=1, lookback=1,test_size=0.3, epochs=100000, batch_size=128, neurons=128, recurrent_dropout=0.08, dropout=0.01, dense=1, loss= "mse", optimiser = "adam", patience=100):
  dataset_data  = datasetLoader("dataset/"+dataset+".csv")
  data = dataset_data.values
  scaler = MinMaxScaler()
  data_scaled  = scaler.fit_transform(data.reshape(-1,1))
  #
  # if sequence_length is not 1: 
  X, y = windowGen(data_scaled, sequence_length,lookback)
  X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size, shuffle=False)
  input_shape = (sequence_length, 1)
  s =dt.datetime.now()
  model = Sequential()
  model.add(LSTM(neurons, input_shape=input_shape, return_sequences=True, recurrent_dropout=recurrent_dropout, dropout=dropout))
  model.add(LSTM(neurons, input_shape=input_shape, return_sequences=True, recurrent_dropout=recurrent_dropout, dropout=dropout))
  model.add(LSTM(neurons, input_shape=input_shape, return_sequences=True, recurrent_dropout=recurrent_dropout, dropout=dropout))
  model.add(LSTM(neurons, input_shape=input_shape, return_sequences=True, recurrent_dropout=recurrent_dropout, dropout=dropout))
  model.add(LSTM(neurons, input_shape=input_shape, return_sequences=True, recurrent_dropout=recurrent_dropout, dropout=dropout))
  model.add(LSTM(neurons, input_shape=input_shape, return_sequences=True, recurrent_dropout=recurrent_dropout, dropout=dropout))
  model.add(LSTM(neurons, input_shape=input_shape, return_sequences=True, recurrent_dropout=recurrent_dropout, dropout=dropout))
  model.add(LSTM(neurons, input_shape=input_shape, return_sequences=True, recurrent_dropout=recurrent_dropout, dropout=dropout))
  model.add(LSTM(neurons, input_shape=input_shape, return_sequences=True, recurrent_dropout=recurrent_dropout, dropout=dropout))
  model.add(LSTM(neurons, input_shape=input_shape, return_sequences=True, recurrent_dropout=recurrent_dropout, dropout=dropout))
  model.add(LSTM(neurons, return_sequences=False, recurrent_dropout=recurrent_dropout, dropout=dropout))
  model.add(Dense(dense))
  model.compile(loss=loss, optimizer=optimiser)
  monitor = EarlyStopping(monitor='loss', patience=patience, verbose=0, mode='auto', restore_best_weights=True)
  history = model.fit(X_train,y_train,validation_data=(X_test,y_test), callbacks=[monitor],verbose=2,epochs=epochs,batch_size=batch_size)#added batchsize
  e =dt.datetime.now()
  time = e-s
  return scaler, model, history , X_train, X_test, y_train, y_test, time