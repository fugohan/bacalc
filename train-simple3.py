# !wandb login d4dd581c6b8708a9029489333664fde8d1fd27d3
# import os
# from google.colab import drive
# drive.mount('/content/drive')
# os.chdir("dr'ive/My Drive/Colab Notebooks")
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
# from sklearn.external.joblib import joblib
import joblib
import datetime as dt 
###
import wandb
from wandb.keras import WandbCallback

dataset = "litecoin"

### Window_Generator 
sequence_length = 1
lookback = 1
### Dataset-Split 

test_size = 0.3

### NN-config 

batch_size = 256
epochs = int(100e3)

neurons = 128
input_shape = (sequence_length,1)
recurrent_dropout = 0.25
dropout = 0.0
dense = 1
loss = "mean_square_error"
optimiser = "adam"

### WANDB Project
project_name = "Experiment0-test3"
####
darkmode = False




def plot_litecoin_res(scaler, model, history , X_train, X_test, y_train, y_test,time):
  # print(os.getcwd())
  testdir = '/test1/11820/'
  # newpath = os.getcwd()+testdir 
  # if not os.path.exists(newpath):
  #     os.makedirs(newpath)

  trainPrediction = model.predict(X_train, batch_size=batch_size)
  testPrediction = model.predict(X_test, batch_size=batch_size)
  plt.plot(history.history["loss"])
  plt.text(0,np.max(history.history["loss"])+np.min(history.history["loss"])*0.1,"min-loss:"+str(float(np.min(history.history["loss"])))+"\nepochs:"+str(len(history.history['loss'])))
  # np.min([1,2,3])
  # plt.savefig(newpath+"loss"+str(i)+".png", dpi=150)
  plt.show()
  TRS =scaler.inverse_transform(trainPrediction)
  YTS =scaler.inverse_transform(y_train)
  # for price in apple['close']:
#     wandb.log({"Stock Price": price})
  i=0
  for i in range(TRS.shape[0]):
    wandb.log({'predicted':TRS[i][0], 'real': YTS[i][0]})
  wandb.log({'pred_all':TRS})
  rms1 = np.sqrt(sklearn.metrics.mean_squared_error(YTS,TRS))
  wandb.log({'RMSE':rms1})
  x1 = 1500
  x2 = 1550
  y1 = np.min(TRS[x1:x2])-1
  y2 = np.max(YTS[x1:x2])+1
  plt.text((x1+1),y2-1,"RMSE="+str(rms1))
  plt.plot(TRS, label="prediction")
  plt.plot(YTS, label="real")
  plt.axis([x1,x2,y1,y2])
  plt.legend()
  # plt.savefig(newpath+"chart"+str(i)+".png", dpi=150)
  wandb.log({"chart": plt})
  plt.show()

  plt.text(0,0,"Time:"+str(time)+"\nConfig:\nneuons:"+str(neurons)+"\ndropout:"+str(dropout)+"\nrecurrent-dropout:"+str(recurrent_dropout)+"\nRMSE:"+str(rms1))
  # plt.axis([-0.01,0.05,-0.01,0.01])
  # plt.plot()
  # plt.savefig(newpath+"config"+str(i)+".png", dpi=150)
  plt.show()
  # return TRS
# plot_litecoin_res(scaler, model, history , X_train, X_test, y_train, y_test, time)


import functions.experiment0 as t
#1
hyperparameter_defaults = dict(
  model = "SimpleRNN",
  dropout = 0.0,
  recurrent_dropout = 0.0,
  dataset = "litecoin",
  sequence_length = 1, 
  neurons = 135,
  batch_size = 100,
  epochs = 2,
  )
wandb.init(
  project=project_name,
  config= hyperparameter_defaults
, sync_tensorboard=True)

config = wandb.config
# print(config.model)
# print(config.dropout)
# print(t.testmodel1())



# if config.model is "SimpleRNN":
#   try:
scaler, model, history, X_train, X_test, y_train, y_test,time = t.testmodel1(dropout=config.dropout, recurrent_dropout=config.recurrent_dropout, epochs=config.epochs, sequence_length=config.sequence_length, neurons=config.neurons, batch_size=config.batch_size, dataset=config.dataset)
plot_litecoin_res(scaler, model, history , X_train, X_test, y_train, y_test,time)
#   except Exception as ex:
#     print(ex)
# if config.model is "LSTM":
#   print("start")
#   scaler, model, history, X_train, X_test, y_train, y_test,time = t.testmodel2(dropout=config.dropout, recurrent_dropout=config.recurrent_dropout, epochs=config.epochs, sequence_length=config.sequence_length, neurons=config.neurons, batch_size=config.batch_size, dataset=config.dataset)
#   plot_litecoin_res(scaler, model, history , X_train, X_test, y_train, y_test,time)
#   print("end")
# if config.model is "GRU":
#   scaler, model, history, X_train, X_test, y_train, y_test,time = t.testmodel3(dropout=config.dropout, recurrent_dropout=config.recurrent_dropout, epochs=config.epochs, sequence_length=config.sequence_length, neurons=config.neurons, batch_size=config.batch_size, dataset=config.dataset)
#   plot_litecoin_res(scaler, model, history , X_train, X_test, y_train, y_test,time)
