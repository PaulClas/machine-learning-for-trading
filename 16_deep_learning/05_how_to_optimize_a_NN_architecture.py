# coding: utf-8

import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

import tensorflow as tf
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping, TensorBoard

np.random.seed(42)

data = pd.read_hdf('data.h5', 'returns')
features, label = data.drop('label', axis=1), data.label
input_dim = features.shape[1]


def auc_roc(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.metrics.auc(y_true, y_pred)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value


def make_model(dense_layers, activation, dropout):
    '''Creates a multi-layer perceptron model
    
    dense_layers: List of layer sizes; one number per layer
    '''

    model = Sequential()
    for i, layer_size in enumerate(dense_layers, 1):
        if i == 1:
            model.add(Dense(layer_size, input_dim=input_dim))
            model.add(Activation(activation))
        else:
            model.add(Dense(layer_size))
            model.add(Activation(activation))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='Adam',
                  metrics=['binary_accuracy', auc_roc])

    return model


test_size, n_splits = .1, 5
X_train, X_test, y_train, y_test = train_test_split(features, label,
                                                    test_size=test_size,
                                                    random_state=42,
                                                    shuffle=True,
                                                    stratify=data.label)

clf = KerasClassifier(make_model, epochs=10, batch_size=32)

cv = StratifiedKFold(n_splits=5, shuffle=True)

param_grid = {'dense_layers': [[32], [32, 32], [64], [64, 64], [64, 64, 32], [64, 32], [128]],
              'activation'  : ['relu', 'tanh'],
              'dropout'     : [.25, .5, .75],
              }

gs = GridSearchCV(estimator=clf,
                  param_grid=param_grid,
                  scoring='roc_auc',
                  cv=cv,
                  refit=True,
                  return_train_score=True,
                  n_jobs=-1,
                  verbose=1,
                  error_score=np.nan
                  )

fit_params = dict(callbacks=[EarlyStopping(monitor='auc_roc', patience=300, verbose=1, mode='max')],
                  verbose=2,
                  epochs=50)

gs.fit(X=X_train.astype(float), y=y_train, **fit_params)
print('\nBest Score: {:.2%}'.format(gs.best_score_))
print('Best Params:\n', pd.Series(gs.best_params_))

dump(gs, 'gs.joblib')
gs.best_estimator_.model.save('best_model.h5')
"""
Best Score: 77.38%
Best Params:
 activation          relu
dense_layers    [64, 64]
dropout              0.5

"""
with pd.HDFStore('data.h5') as store:
    store.put('X_train', X_train)
    store.put('X_test', X_test)
    store.put('y_train', y_train)
    store.put('y_test', y_test)
