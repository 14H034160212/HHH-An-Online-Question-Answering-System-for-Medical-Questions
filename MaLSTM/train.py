from time import time
import pandas as pd

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf
# tf.enable_eager_execution()

from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input, Embedding, LSTM, GRU, Conv1D, Conv2D, GlobalMaxPool1D, Dense, Dropout

from util import make_w2v_embeddings
from util import split_and_zero_padding
from util import ManDist
from tensorflow.keras.callbacks import TensorBoard
from tensorboard.plugins import projector
import os

# File paths
TRAIN_CSV = '/root/HHH-An-Online-Question-Answering-System-for-Medical-Questions/Data/Model_train_dev_test_dataset/Other_model_train_dev_test_dataset/train.csv'

# Load training set
train_df = pd.read_csv(TRAIN_CSV,encoding = 'gb18030')
for q in ['question1', 'question2']:
    train_df[q + '_n'] = train_df[q]

# Make word2vec embeddings
embedding_dim = 300
max_seq_length = 20
use_w2v = True

train_df, embeddings = make_w2v_embeddings(train_df, embedding_dim=embedding_dim, empty_w2v=not use_w2v)

# Split to train validation
validation_size = int(len(train_df) * 0.1)
training_size = len(train_df) - validation_size

X = train_df[['question1_n', 'question2_n']]
Y = train_df['is_duplicate']

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size)

X_train = split_and_zero_padding(X_train, max_seq_length)
X_validation = split_and_zero_padding(X_validation, max_seq_length)

# Convert labels to their numpy representations
Y_train = Y_train.values
Y_validation = Y_validation.values

# Make sure everything is ok
assert X_train['left'].shape == X_train['right'].shape
assert len(X_train['left']) == len(Y_train)

# --

# Model variables
gpus = 1
batch_size = 1024 * gpus
n_epoch = 9
n_hidden = 50

# Define the shared model
x = Sequential()
x.add(Embedding(len(embeddings), embedding_dim,
                weights=[embeddings], input_shape=(max_seq_length,), trainable=False))
# CNN
# x.add(Conv1D(250, kernel_size=5, activation='relu'))
# x.add(GlobalMaxPool1D())
# x.add(Dense(250, activation='relu'))
# x.add(Dropout(0.3))
# x.add(Dense(1, activation='sigmoid'))
# LSTM
x.add(LSTM(n_hidden))

shared_model = x

# The visible layer
left_input = Input(shape=(max_seq_length,), dtype='int32')
right_input = Input(shape=(max_seq_length,), dtype='int32')

# Pack it all up into a Manhattan Distance model
malstm_distance = ManDist()([shared_model(left_input), shared_model(right_input)])
model = Model(inputs=[left_input, right_input], outputs=[malstm_distance])

#if gpus >= 2:
    # `multi_gpu_model()` is a so quite buggy. it breaks the saved model.
#    model = tf.keras.utils.multi_gpu_model(model, gpus=gpus)
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
model.summary()
shared_model.summary()

# Start trainings
training_start_time = time()
logdir = "./MaLSTM/logs"
if not os.path.exists(logdir):
  os.makedirs(logdir)

with open(os.path.join(logdir, 'metadata.tsv'), "w") as f:
  for left in train_df['question1'].tolist():
    f.write("{}\n".format(left))
  for right in train_df['question2'].tolist():
    f.write("{}\n".format(right))
    
tensorboard = TensorBoard(log_dir = logdir,
                            histogram_freq = 0,
                            write_graph = True,
                            write_images = False,
                            # embeddings_freq = 1,
                            update_freq='epoch',
                            embeddings_layer_names = None,
                            embeddings_data = [X_validation['left'], X_validation['right']])
    
malstm_trained = model.fit([X_train['left'], X_train['right']], Y_train,
                           batch_size=batch_size, epochs=n_epoch,
                           validation_data=([X_validation['left'], X_validation['right']], Y_validation),
                           verbose=1,
                           callbacks=[tensorboard])

training_end_time = time()

weights = tf.Variable(model.layers[2].get_weights()[0][1:])
checkpoint = tf.train.Checkpoint(embedding=weights)
checkpoint.save(os.path.join(logdir, "embedding.ckpt"))

# Set up config.
config = projector.ProjectorConfig()
embedding = config.embeddings.add()
# The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`.
embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
embedding.metadata_path = 'metadata.tsv'
projector.visualize_embeddings(logdir, config)

tf.train.Checkpoint(embedding=weights)

# # Initialize a TensorFlow session
# with tf.compat.v1.Session() as sess:
#     sess.run(tf.compat.v1.global_variables_initializer())

#     # Create a TensorFlow summary writer
#     summary_writer = tf.compat.v1.summary.FileWriter(logdir)

#     # Configure the projector
#     config = projector.ProjectorConfig()
#     embedding = config.embeddings.add()
#     embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
#     embedding.metadata_path = 'metadata.tsv'

#     # Save the projector configuration
#     projector.visualize_embeddings(summary_writer, config)
    
#     tf.train.Checkpoint(embedding=weights)
    
#     # tf.compat.v1.train.Saver(sess, os.path.join(logdir, "model_embedding.ckpt"))

#     # # Write the embeddings to the summary writer
#     # summary_writer.add_embedding(weights,metadata=None)

#     # Close the summary writer
#     summary_writer.close()

print("Training time finished.\n%d epochs in %12.2f" % (n_epoch,
                                                        training_end_time - training_start_time))

model.save('./MaLSTM/SiameseLSTM.h5')

# Plot accuracy
plt.subplot(211)
plt.plot(malstm_trained.history['accuracy'])
plt.plot(malstm_trained.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot loss
plt.subplot(212)
plt.plot(malstm_trained.history['loss'])
plt.plot(malstm_trained.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.tight_layout(h_pad=1.0)
plt.savefig('./MaLSTM/history-graph.png')

print(str(malstm_trained.history['val_accuracy'][-1])[:6] +
      "(max: " + str(max(malstm_trained.history['val_accuracy']))[:6] + ")")
print("Done.")
