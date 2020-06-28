
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # specify gpu 0 or 1
import numpy as np
import tensorflow as tf
import random as rn
import keras.backend as K

np.random.seed(123)
rn.seed(123)
tf.set_random_seed(123)


import pandas as pd
from keras import regularizers
from keras.layers import Dense
from scipy.stats import pearsonr
from keras.optimizers import Adam
from keras.layers import Dropout
from keras.models import Sequential
#import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
#from keras.wrappers.scikit_learn import KerasRegressor
from KerasRegressor_Custom import Custom_KerasRegressor
from keras.backend.tensorflow_backend import set_session
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from keras.models import load_model

def relu_max100(x):
    return K.relu(x, max_value=100)

config=tf.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))

data = pd.read_csv('/Data/Suzuki_Miyaura_model_scale.csv', sep=',')
# 4. Split data into training and test sets
#print(data)
Y = data.Yield
X = data.drop([ 'Yield', 'catalyst','Scheme'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(
                         X, Y, test_size=0.2, random_state=1)


def create_model(dropout_1=0.0,
                 dropout_2=0.0,
                 learning_rate=0.0
                 ):
    # create model
    model = Sequential()
    model.add(
        Dense(
            1000,
            input_dim=44,
            kernel_regularizer=regularizers.l2(0.001),
            activity_regularizer=regularizers.l2(0.001),
            bias_regularizer=regularizers.l2(0.001),
            kernel_initializer='normal',
            activation='relu'))
    model.add(Dropout(dropout_1))
    
    model.add(
        Dense(
            1000,
            kernel_regularizer=regularizers.l2(0.001),
            activity_regularizer=regularizers.l2(0.001),
            bias_regularizer=regularizers.l2(0.001),
            kernel_initializer='normal',
            activation='relu'))
    model.add(Dropout(dropout_2))
    

    
    model.add(Dense(1, kernel_initializer='normal', activation=relu_max100))
    adam = Adam(lr=learning_rate, decay=0.00001)
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mae'])
    return model


model = create_model()
model = Custom_KerasRegressor(build_fn=create_model, verbose=0)
# grid search epochs, batch size and optimizer
#nodes_1 = [100,500]
#nodes_2 = [500]
dropout_1 = [0.1,0.2,0.3,0.5]
dropout_2 = [0.1,0.2,0.3,0.5]
learning_rate = [0.001,0.0001]
batches = [5,10,20]
param_grid = dict(
    dropout_1=dropout_1,dropout_2=dropout_2,learning_rate=learning_rate,batch_size=batches)
model = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5)
earlyStopping = EarlyStopping(
    monitor='val_loss', patience=20, min_delta=0.01, verbose=0, mode='min')
model.fit(x_train, y_train, epochs=500, verbose=1, callbacks=[earlyStopping])

print(model.grid_scores_)
print(model.best_params_)

h5_modelfile = "/Model/DNN_model_best.h5"

model.best_estimator_.model.save(h5_modelfile)
#del model
model = load_model(h5_modelfile, custom_objects={'relu_max100': relu_max100})

y_pred = model.predict(x_test)
y_pred = y_pred.ravel()
y_pred = y_pred.tolist()


def regressor_metrics(y, y_pred):
    score = {
        'MAE': mean_absolute_error(y, y_pred),
        'MSE': mean_squared_error(y, y_pred),
        'r2_score': r2_score(y, y_pred),
        'R2_score': ((pearsonr(y, y_pred))[0])**2
    }
    RMSE = np.sqrt(mean_squared_error(y, y_pred))
    return score, RMSE


print(regressor_metrics(y_test, y_pred))
test_results = pd.DataFrame({
    'Predicted Yield'


: y_pred,
    'Observed Yield': y_test
})
test_results.to_csv(
    '/Result/DNN_test_results.csv', index=False)
x_pred = model.predict(x_train)
x_pred=x_pred.ravel()
x_pred=x_pred.tolist()
print(regressor_metrics(y_train, x_pred))
train_results = pd.DataFrame({
    'Predicted Yield': x_pred,
    'Observed Yield': y_train
})
train_results.to_csv(
    '/Result/DNN_train_results.csv', index=False)


external=pd.read_csv('/Data/Suzuki_Miyaura_external_scale.csv',sep=',')
external_data=external.drop(['Scheme','catalyst','Yield'],axis=1)
external_yield=external.Yield
external_pred=model.predict(external_data)
external_pred=external_pred.ravel()
external_pred=external_pred.tolist()
print(regressor_metrics(external_yield,external_pred))

external_results=pd.DataFrame({
    'Predicted Yield': external_pred,
    'Observed Yield': external_yield
})
external_results.to_csv('/Result/DNN_external_results.csv',index=False)


scanning=pd.read_csv('/Data/Suzuki_Miyaura_scanning_scale.csv',sep=',')
scanning_data=scanning.drop(['Scheme','catalyst'],axis=1)
scanning_yield=model.predict(scanning_data)
scanning['Predicted_Yield']=scanning_yield
scanning.to_csv('/Result/DNN_scanning_results.csv',index=False)

cross=pd.read_csv('/Data/Suzuki_Miyaura_cross_scale.csv',sep=',')
cross_data=cross.drop(['experiment','catalyst'],axis=1)
cross_yield=model.predict(cross_data)
cross['Predicted_Yield']=cross_yield
cross.to_csv('/Result/DNN_cross_results.csv',index=False)

new=pd.read_csv('/Data/Suzuki_Miyaura_new_scale.csv',sep=',')
new_data=new.drop(['Scheme','catalyst'],axis=1)
new_yield=model.predict(new_data)
new['Predicted_Yield']=new_yield
new.to_csv('/Result/DNN_new_results.csv',index=False)