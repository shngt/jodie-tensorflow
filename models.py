import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
#import torch
#import gpustat
from collections import defaultdict

# SELECT THE GPU WITH MOST FREE MEMORY TO SCHEDULE JOB 
#def select_free_gpu():
#    mem = []
#    # Seems sort of wasteful to import torch just for this. Need to review if there's a possible fix
#    gpus = list(set(range(torch.cuda.device_count()))) # list(set(X)) is done to shuffle the array
#    for i in gpus:
#        gpu_stats = gpustat.GPUStatCollection.new_query()
#        mem.append(gpu_stats.jsonify()["gpus"][i]["memory.used"])
#    return str(gpus[np.argmin(mem)])

class JODIE(tf.keras.Model):
    def __init__(self, args, num_features, num_users, num_items, **kwargs):
        print("Initialising the JODIE model")
        super().__init__(name=args.model, **kwargs)
        self.embedding_dim = args.embedding_dim
        self.num_users = num_users
        self.num_items = num_items
        self.user_static_embedding_size = num_users
        self.item_static_embedding_size = num_items
        self.reg = keras.regularizers.l2(1e-5)

        print("Initialising RNN inputs")
        item_rnn_input_size = user_rnn_input_size = self.embedding_dim + 1 + num_features
        #self.item_rnn_input = keras.Input(shape=(item_rnn_input_size,))
        self.item_rnn_input_reshaped = keras.layers.Reshape((1, item_rnn_input_size))
        #self.item_rnn_hidden_state_reshaped = keras.layers.Reshape((1, self.embedding_dim))

        #self.user_rnn_input = keras.Input(shape=(user_rnn_input_size,))
        self.user_rnn_input_reshaped = keras.layers.Reshape((1, user_rnn_input_size))
        #self.user_rnn_hidden_state_reshaped = keras.layers.Reshape((1, self.embedding_dim))

        print("Initialising user and item RNN")
        self.item_rnn_layer = keras.layers.GRU(units=self.embedding_dim, kernel_regularizer=self.reg, recurrent_regularizer=self.reg, bias_regularizer=self.reg)
        self.item_rnn_layer_normed = keras.layers.LayerNormalization()
                
        self.user_rnn_layer = keras.layers.GRU(units=self.embedding_dim, kernel_regularizer=self.reg, recurrent_regularizer=self.reg, bias_regularizer=self.reg)
        self.user_rnn_layer_normed = keras.layers.LayerNormalization()

        print("Initialising linear layers")
        self.linear_layer_1 = keras.layers.Dense(units=50, activation=tf.nn.relu, kernel_regularizer=self.reg, bias_regularizer=self.reg)
        self.linear_layer_2 = keras.layers.Dense(units=2, kernel_regularizer=self.reg, bias_regularizer=self.reg)
        self.prediction_layer = keras.layers.Dense(units=self.item_static_embedding_size + self.embedding_dim, kernel_regularizer=self.reg, bias_regularizer=self.reg)
        self.embedding_layer = keras.layers.Dense(units=self.embedding_dim, kernel_initializer=keras.initializers.RandomNormal(stddev=1/self.embedding_dim), bias_initializer=keras.initializers.RandomNormal(stddev=1/self.embedding_dim), kernel_regularizer=self.reg, bias_regularizer=self.reg)

    def call(self, user_embeddings, item_embeddings, timediffs=None, features=None, select=None):
        if select is "item_update":
            item_rnn_input = keras.layers.Concatenate()([user_embeddings, timediffs, features])
            #item_embeddings_reshaped = self.user_rnn_hidden_state_reshaped(item_embeddings)
            x = self.item_rnn_input_reshaped(item_rnn_input)
            y = self.item_rnn_layer(x, initial_state=item_embeddings)
            z = self.item_rnn_layer_normed(y)
            return z
        
        elif select is "user_update":
            user_rnn_input = keras.layers.Concatenate()([item_embeddings, timediffs, features])
            #user_embeddings_reshaped = self.user_rnn_hidden_state_reshaped(user_embeddings)
            x = self.user_rnn_input_reshaped(user_rnn_input)
            y = self.user_rnn_layer(x, initial_state=user_embeddings)
            z = self.item_rnn_layer_normed(y)
            return z
        
        elif select == 'project':
            user_projected_embedding = self.context_convert(user_embeddings, timediffs, features)
            return user_projected_embedding

    def context_convert(self, embeddings, timediffs, features):
        new_embeddings = tf.math.multiply(embeddings, tf.math.add(1.0, self.embedding_layer(timediffs)))
        return new_embeddings

    def predict_label(self, user_embeddings):
        x = self.linear_layer_1(user_embeddings)
        X_out = self.linear_layer_2(x)
        return X_out

    def predict_item_embedding(self, user_embeddings):
        X_out = self.prediction_layer(user_embeddings)
        return X_out

# INITIALIZE T-BATCH VARIABLES
def reinitialize_tbatches():
    global current_tbatches_interactionids, current_tbatches_user, current_tbatches_item, current_tbatches_timestamp, current_tbatches_feature, current_tbatches_label, current_tbatches_previous_item
    global tbatchid_user, tbatchid_item, current_tbatches_user_timediffs, current_tbatches_item_timediffs, current_tbatches_user_timediffs_next

    # list of users of each tbatch up to now
    current_tbatches_interactionids = defaultdict(list)
    current_tbatches_user = defaultdict(list)
    current_tbatches_item = defaultdict(list)
    current_tbatches_timestamp = defaultdict(list)
    current_tbatches_feature = defaultdict(list)
    current_tbatches_label = defaultdict(list)
    current_tbatches_previous_item = defaultdict(list)
    current_tbatches_user_timediffs = defaultdict(list)
    current_tbatches_item_timediffs = defaultdict(list)
    current_tbatches_user_timediffs_next = defaultdict(list)

    # the latest tbatch a user is in
    tbatchid_user = defaultdict(lambda: -1)

    # the latest tbatch a item is in
    tbatchid_item = defaultdict(lambda: -1)

    #total_reinitialization_count +=1

# CALCULATE LOSS FOR THE PREDICTED USER STATE 
def calculate_state_prediction_loss(model, tbatch_interactionids, user_embeddings_time_series, y_true, loss_function):
    # PREDICT THE LABEL FROM THE USER DYNAMIC EMBEDDINGS
    prob = model.predict_label(user_embeddings_time_series[tbatch_interactionids, :])
    y = tf.Variable(y_true[tbatch_interactionids], tf.int64)
    loss = loss_function(prob, y)
    return loss