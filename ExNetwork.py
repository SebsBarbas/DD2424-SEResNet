import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from LoadDataset import *
from ResNet50 import *

classes = 10
img_size = 32
img_channels = 3

learning_rate = 0.01
reduction_ratio = 3

batch_size = 128
epochs = 10
iterations = 100


# Variables
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
test_top1 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1, name='test_1_accuracy') 
test_top5 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='test_5_accuracy') 

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training = True)
        loss =loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training = False)
    t_loss =loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)
	test_top1(labels, predictions)
	test_top5(labels, predictions)


if __name__ == "__main__":

    save_model_dir = ".\checkpoints"    

    #Load data
    print("loading data...")
    train_X, train_lab, test_X, test_lab = get_data()
    print("normalizing data...")
    train_X, test_X = normalize(train_X, test_X)

    data = tf.placeholder(tf.float32, shape=[None, img_size, img_size, img_channels])
    model = ResNet50(include_top=True, input_tensor=data, input_shape=[None, img_size, img_size, img_channels], pooling=None, classes=classes)


    checkpoint_dir = os.path.join(save_model_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restaurado de {}".format(manager.latest_checkpoint))
    else:
        print("Inicializando desde cero")
    #saver = tf.train.Saver(tf.global_variables())
    #ckpt = tf.train.get_checkpoint_state('./checkpoints')

    losses = []
    accs = []
    for i in range(epoch):
        img_ind = 0

        for step in range(iterations):
            if img_ind + batch_size < train_X.get_shape()[0]:
                batch_X = train_X[img_ind : img_ind + batch_size]
                batch_Y = train_lab[img_ind : img_ind + batch_size]
            else:
                batch_X = train_X[img_ind:]
                batch_Y = train_lab[img_ind:]

            train_step(batch_X, batch_Y)

            template = 'Epoch: {}, Iteration: {}, Loss: {}, Accuracy: {}'
            print(template.format(i+1, step+1,
                            train_loss.result(),
                            train_accuracy.result()*100)) 

            save_path = manager.save()

            losses.append(train_loss.result())
            accs.append(train_accuracy.result())

            train_loss.reset_states()
            train_accuracy.reset_states()

    test_step(test_X, test_lab)
	template = 'Loss: {}, Accuracy: {}, Top1 Error: {}, Top5 Error: {}'
            print(template.format(train_loss.result(),
                            train_accuracy.result()*100,
							(1 - test_top1.result())*100,
							(1 - test_top5.result())*100))


"""							
%load_ext tensorboard		
import datetime

!rm -rf ./logs/

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

for epoch in range(EPOCHS):
  for (x_train, y_train) in train_dataset:
    train_step(model, optimizer, x_train, y_train)
  with train_summary_writer.as_default():
    tf.summary.scalar('loss', train_loss.result(), step=epoch)
    tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

  for (x_test, y_test) in test_dataset:
    test_step(model, x_test, y_test)
  with test_summary_writer.as_default():
    tf.summary.scalar('loss', test_loss.result(), step=epoch)
    tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)
  
  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
  print (template.format(epoch+1,
                         train_loss.result(), 
                         train_accuracy.result()*100,
                         test_loss.result(), 
                         test_accuracy.result()*100))

  # Reset metrics every epoch
  train_loss.reset_states()
  test_loss.reset_states()
  train_accuracy.reset_states()
  test_accuracy.reset_states()
  
  #run in the command line: tensorboard --logdir logs/fit
  """