from __future__ import absolute_import, division, print_function
from tensorflow import keras
import numpy as np
import flask
from flask import Flask, render_template, request
import pandas as pd
import time
import tensorflow as tf
from keras.utils.training_utils import multi_gpu_model


print(tf.__version__)
print(flask.__version__)
print(np.__version__)
print(pd.__version__)

app = Flask(__name__)
boston = keras.datasets.boston_housing
tf_gpus = 0


def load_and_train_model():
    print('Training model')
    (train_data, train_labels), (test_data, test_labels) = boston.load_data()

    # Shuffle the training set
    order = np.argsort(np.random.random(train_labels.shape))
    train_data = train_data[order]
    train_labels = train_labels[order]

    print("Training set: {}".format(train_data.shape))  # 404 examples, 13 features
    print("Testing set:  {}".format(test_data.shape))  # 102 examples, 13 features

    model = build_model(train_data, tf_gpus)
    model.summary()

    EPOCHS = 500

    # Store training stats
    start = time.time()
    history = model.fit(train_data, train_labels, epochs=EPOCHS,
                        validation_split=0.2, verbose=0,
                        callbacks=[PrintDot()])
    print('\nElapsed time: {}s\n'.format(time.time() - start))

    test_model(test_data, test_labels, model)
    return model


# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')


def build_model(train_data, gpus):
    print(train_data.shape[1])
    model = keras.Sequential([
        keras.layers.Dense(64, activation=tf.nn.relu, input_shape=(train_data.shape[1],)),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(1)
    ])

    with tf.device("/cpu:0"):
        model.build(train_data.shape)

    # make the model parallel
    if gpus > 1:
        model = multi_gpu_model(model, gpus=gpus)

    optimizer = tf.train.RMSPropOptimizer(0.001)
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae'])
    return model


def test_model(test_data, test_labels, model):
    print('Testing model')
    [loss, mae] = model.evaluate(test_data, test_labels, verbose=0)
    print("Testing set Mean Abs Error: ${:7.2f}".format(mae * 1000))
    print("Loss: {}".format(loss))
    test_predictions = model.predict(test_data[0:1]).flatten()
    print(test_predictions)


# REST API
@app.route("/predict", methods=["POST"])
def predict():
    post_data = request.get_json()
    # print('post_data=', post_data)

    params = [[float(post_data['CRIM']),
              float(post_data['ZN']),
              float(post_data['INDUS']),
              float(post_data['CHAS']),
              float(post_data['NOX']),
              float(post_data['RM']),
              float(post_data['AGE']),
              float(post_data['DIS']),
              float(post_data['RAD']),
              float(post_data['TAX']),
              float(post_data['PTRATIO']),
              356.674032,
              float(post_data['LSTAT'])]]

    np_params = np.asarray(params)
    # print(np_params.shape)
    # print('params array=', np_params) # DEBUG

    # resp_data = {"estimate": 0.00}
    price_est = model.predict(np_params)
    print('THE ESTIMATE IS:', price_est) # DEBUG
    resp_data = {"estimate": format(price_est[0][0] * 1000, '.2f')}  # default response
    return flask.jsonify(resp_data)


@app.route("/test", methods=["GET"])
def test_api():

    print('This is a test.') # DEBUG
    resp_data = {'status': 'success'}
    return flask.jsonify(resp_data)


# JINJA TEMPLATES
@app.route('/')
def form():
    return render_template('input.html')


if __name__ == "__main__":
    print("* Loading model and Flask starting server... please wait.")
    model = load_and_train_model()
    app.run(host='0.0.0.0', port=80, debug=False)

