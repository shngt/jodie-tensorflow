import argparse
import os
import sys
from tqdm import trange
import models as lib
from models import *
from utils import *
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
#K.set_floatx('float64')

# INITIALIZE PARAMETERS
parser = argparse.ArgumentParser()
parser.add_argument('--network', required=True, help='Name of the network/dataset')
parser.add_argument('--model', default="jodie", help='Model name to save output in file')
# parser.add_argument('--gpu', default=-1, type=int, help='ID of the gpu to run on. If set to -1 (default), the GPU with most free memory will be chosen.')
parser.add_argument('--epochs', default=50, type=int, help='Number of epochs to train the model')
parser.add_argument('--embedding_dim', default=128, type=int, help='Number of dimensions of the dynamic embedding')
parser.add_argument('--train_proportion', default=0.8, type=float, help='Fraction of interactions (from the beginning) that are used for training.The next 10% are used for validation and the next 10% for testing')
parser.add_argument('--state_change', default=True, type=bool, help='True if training with state change of users along with interaction prediction. False otherwise. By default, set to True.') 
args = parser.parse_args()

args.datapath = "data/%s.csv" % args.network 
if args.train_proportion > 0.8:
    sys.exit('Training sequence proportion cannot be greater than 0.8.')

# SET GPU
#if args.gpu == -1:
#    args.gpu = select_free_gpu()
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

[user2id, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence,
 item2id, item_sequence_id, item_timediffs_sequence, 
 timestamp_sequence, feature_sequence, y_true] = load_network(args)
num_interactions = len(user_sequence_id)
num_users = len(user2id) 
num_items = len(item2id) + 1 # one extra item for "none-of-these"
num_features = len(feature_sequence[0])
true_labels_ratio = len(y_true)/(1.0+sum(y_true)) # +1 in denominator in case there are no state change labels, which will throw an error. 
print("*** Network statistics:\n  %d users\n  %d items\n  %d interactions\n  %d/%d true labels ***\n\n" % (num_users, num_items, num_interactions, sum(y_true), len(y_true)))

# SET TRAINING, VALIDATION, TESTING, and TBATCH BOUNDARIES
train_end_idx = validation_start_idx = int(num_interactions * args.train_proportion) 
test_start_idx = int(num_interactions * (args.train_proportion+0.1))
test_end_idx = int(num_interactions * (args.train_proportion+0.2))

# SET BATCHING TIMESPAN
'''
Timespan is the frequency at which the batches are created and the JODIE model is trained. 
As the data arrives in a temporal order, the interactions within a timespan are added into batches (using the T-batch algorithm). 
The batches are then used to train JODIE. 
Longer timespans mean more interactions are processed and the training time is reduced, however it requires more GPU memory.
Longer timespan leads to less frequent model updates. 
'''
timespan = timestamp_sequence[-1] - timestamp_sequence[0]
tbatch_timespan = timespan / 500 
#reinitialize_tbatches.total_reinitialization_count = 0

model = JODIE(args, num_features, num_users, num_items)
weight = tf.constant([1, true_labels_ratio])
crossEntropyLoss = tf.nn.weighted_cross_entropy_with_logits
MSELoss = keras.losses.MeanSquaredError()

# INITIALIZE MODEL
learning_rate = 1e-3
optimizer = keras.optimizers.Adam(learning_rate)

# INITIALIZE EMBEDDING
initial_user_embedding = tf.Variable(K.l2_normalize(K.random_uniform((1, args.embedding_dim)))) # the initial user and item embeddings are learned during training as well
initial_item_embedding = tf.Variable(K.l2_normalize(K.random_uniform((1, args.embedding_dim))))

user_embeddings = K.repeat_elements(initial_user_embedding, num_users, 0) # initialize all users to the same embedding 
item_embeddings = K.repeat_elements(initial_item_embedding, num_users, 0) # initialize all items to the same embedding
item_embedding_static = tf.Variable(K.eye(num_items)) # one-hot vectors for static embeddings
user_embedding_static = tf.Variable(K.eye(num_users)) # one-hot vectors for static embeddings 

# RUN THE JODIE MODEL
'''
THE MODEL IS TRAINED FOR SEVERAL EPOCHS. IN EACH EPOCH, JODIES USES THE TRAINING SET OF INTERACTIONS TO UPDATE ITS PARAMETERS.
'''
print("*** Training the JODIE model for %d epochs ***" % args.epochs)

with trange(args.epochs) as progress_bar_1:
    for ep in progress_bar_1:
        progress_bar_1.set_description('Epoch %d of %d' % (ep, args.epochs))
        # INITIALIZE EMBEDDING TRAJECTORY STORAGE
        user_embeddings_timeseries = tf.Variable(tf.zeros([num_interactions, args.embedding_dim], tf.float32))
        item_embeddings_timeseries = tf.Variable(tf.zeros([num_interactions, args.embedding_dim], tf.float32))

        reinitialize_tbatches()
        total_loss, loss, total_interaction_count = 0, 0, 0

        tbatch_start_time = None
        tbatch_to_insert = -1
        tbatch_full = False

        # TRAIN TILL THE END OF TRAINING INTERACTION IDX
        with trange(train_end_idx) as progress_bar_2:
            for j in progress_bar_2:
                progress_bar_2.set_description('Processed %dth interactions' % j)

                    # READ INTERACTION J
                userid = user_sequence_id[j]
                itemid = item_sequence_id[j]
                feature = feature_sequence[j]
                user_timediff = user_timediffs_sequence[j]
                item_timediff = item_timediffs_sequence[j]

                # CREATE T-BATCHES: ADD INTERACTION J TO THE CORRECT T-BATCH
                tbatch_to_insert = max(lib.tbatchid_user[userid], lib.tbatchid_item[itemid]) + 1 
                lib.tbatchid_user[userid] = tbatch_to_insert 
                lib.tbatchid_item[itemid] = tbatch_to_insert

                lib.current_tbatches_user[tbatch_to_insert].append(userid)
                lib.current_tbatches_item[tbatch_to_insert].append(itemid)
                lib.current_tbatches_feature[tbatch_to_insert].append(feature)
                lib.current_tbatches_interactionids[tbatch_to_insert].append(j)
                lib.current_tbatches_user_timediffs[tbatch_to_insert].append(user_timediff)
                lib.current_tbatches_item_timediffs[tbatch_to_insert].append(item_timediff)
                lib.current_tbatches_previous_item[tbatch_to_insert].append(user_previous_itemid_sequence[j])

                timestamp = timestamp_sequence[j]
                if tbatch_start_time is None:
                    tbatch_start_time = timestamp

                # AFTER ALL INTERACTIONS IN THE TIMESPAN ARE CONVERTED TO T-BATCHES, FORWARD PASS TO CREATE EMBEDDING TRAJECTORIES AND CALCULATE PREDICTION LOSS
                if timestamp - tbatch_start_time > tbatch_timespan:
                    with tf.GradientTape() as tape:
                        tbatch_start_time = timestamp # RESET START TIME FOR THE NEXT TBATCHES

                        # ITERATE OVER ALL T-BATCHES
                        with trange(len(lib.current_tbatches_user)) as progress_bar_3:
                            for i in progress_bar_3:
                                progress_bar_3.set_description('Processed %d of %d T-batches ' % (i, len(lib.current_tbatches_user)))
                                
                                total_interaction_count += len(lib.current_tbatches_interactionids[i])

                                # LOAD THE CURRENT TBATCH
                                tbatch_userids = tf.constant(lib.current_tbatches_user[i], tf.int64) # Recall "lib.current_tbatches_user[i]" has unique elements
                                tbatch_itemids = tf.constant(lib.current_tbatches_item[i], tf.int64) # Recall "lib.current_tbatches_item[i]" has unique elements
                                tbatch_interactionids = tf.constant(lib.current_tbatches_interactionids[i], tf.int64) 
                                feature_tensor = tf.Variable(lib.current_tbatches_feature[i], tf.float32) # Recall "lib.current_tbatches_feature[i]" is list of list, so "feature_tensor" is a 2-d tensor
                                user_timediffs_tensor = tf.reshape(tf.Variable(lib.current_tbatches_user_timediffs[i], tf.float32), [-1, 1])
                                item_timediffs_tensor = tf.reshape(tf.Variable(lib.current_tbatches_item_timediffs[i], tf.float32), [-1, 1])
                                tbatch_itemids_previous = tf.constant(lib.current_tbatches_previous_item[i], tf.int64)
                                item_embedding_previous = tf.gather(item_embeddings, tbatch_itemids_previous)

                                # PROJECT USER EMBEDDING TO CURRENT TIME
                                user_embedding_input = tf.gather(user_embeddings, tbatch_userids)
                                user_projected_embedding = model.call(user_embedding_input, item_embedding_previous, timediffs=user_timediffs_tensor, features=feature_tensor, select='project')
                                user_item_embedding = tf.keras.layers.concatenate([user_projected_embedding, item_embedding_previous, tf.gather(item_embedding_static, tbatch_itemids_previous), tf.gather(user_embedding_static, tbatch_userids)])

                                # PREDICT NEXT ITEM EMBEDDING                            
                                predicted_item_embedding = model.predict_item_embedding(user_item_embedding)

                                # CALCULATE PREDICTION LOSS
                                item_embedding_input = tf.gather(item_embeddings, tbatch_itemids)
                                loss += MSELoss(tf.keras.layers.concatenate([item_embedding_input, tf.gather(item_embedding_static, tbatch_itemids)]), predicted_item_embedding)

                                # UPDATE DYNAMIC EMBEDDINGS AFTER INTERACTION
                                user_embedding_output = model.call(user_embedding_input, item_embedding_input, timediffs=user_timediffs_tensor, features=feature_tensor, select='user_update')
                                item_embedding_output = model.call(user_embedding_input, item_embedding_input, timediffs=item_timediffs_tensor, features=feature_tensor, select='item_update')

                                item_embeddings = item_embeddings.numpy()
                                item_embeddings[tbatch_itemids, :] = item_embedding_output.numpy()
                                item_embeddings = tf.Variable(item_embeddings)
                                user_embeddings = user_embeddings.numpy()
                                user_embeddings[tbatch_userids, :] = user_embedding_output.numpy()
                                user_embeddings = tf.Variable(user_embeddings)

                                user_embeddings_timeseries = user_embeddings_timeseries.numpy()
                                user_embeddings_timeseries[tbatch_interactionids, :] = user_embedding_output.numpy()
                                user_embeddings_timeseries = tf.Variable(user_embeddings_timeseries)
                                item_embeddings_timeseries = item_embeddings_timeseries.numpy()
                                item_embeddings_timeseries[tbatch_interactionids, :] = item_embedding_output.numpy()
                                item_embeddings_timeseries = tf.Variable(item_embeddings_timeseries)

                                # CALCULATE LOSS TO MAINTAIN TEMPORAL SMOOTHNESS
                                loss += MSELoss(item_embedding_input, item_embedding_output)
                                loss += MSELoss(user_embedding_input, user_embedding_output)

                        # BACKPROPAGATE ERROR AFTER END OF T-BATCH
                        total_loss += loss
                        grads = tape.gradient(loss, model.trainable_weights)
                        optimizer.apply_gradients(zip(grads, model.trainable_weights))

                        # RESET LOSS FOR NEXT T-BATCH
                        loss = 0
                        
                        # REINITIALIZE
                        reinitialize_tbatches()
                        tbatch_to_insert = -1

        # END OF ONE EPOCH 
        print("\n\nTotal loss in this epoch = %f" % (total_loss))
        item_embeddings_dystat = K.concatenate([item_embeddings, item_embedding_static])
        user_embeddings_dystat = K.concatenate([user_embeddings, user_embedding_static])
        # SAVE CURRENT MODEL TO DISK TO BE USED IN EVALUATION.
        #save_model(model, optimizer, args, ep, user_embeddings_dystat, item_embeddings_dystat, train_end_idx, user_embeddings_timeseries, item_embeddings_timeseries)

        user_embeddings = K.repeat_elements(initial_user_embedding, num_users, 0) # initialize all users to the same embedding 
        item_embeddings = K.repeat_elements(initial_item_embedding, num_users, 0) # initialize all items to the same embedding


# END OF ALL EPOCHS. SAVE FINAL MODEL DISK TO BE USED IN EVALUATION.
#print "\n\n*** Training complete. Saving final model. ***\n\n"
#save_model(model, optimizer, args, ep, user_embeddings_dystat, item_embeddings_dystat, train_end_idx, user_embeddings_timeseries, item_embeddings_timeseries)












        
