import tensorflow as tf
import numpy as np
import argparse
from PIL import Image
from time import clock
import tables

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save_model", type=int, default=-1,
    help="(optional) whether or not model should be saved to disk")
ap.add_argument("-l", "--load_model", type=int, default=-1,
    help="(optional) whether or not pre-trained model should be loaded")
ap.add_argument("-w", "--model_path", type=str,
    help="(optional) path to weights file")
ap.add_argument("-t", "--test_mode", type=int, default=-1,
    help="(optional) whether you are testing a prediction of a single image")
ap.add_argument("-i", "--image_path", type=str,
    help="(optional) path to the image if you are using test mode" )
args = vars(ap.parse_args())

data_path = "./trainData_32.hdf5"
label_path = "./trainLabel.hdf5"

# declare some hyperparameters
batch_size = 1000
image_height = 32
image_width = 32
num_channels = 8
patch_size = 3
depth = (50, 100, 150, 200, 250, 300, 350, 400)
num_hidden = (900, 200)
keep_prob = (1.0, 0.9, 0.9, 0.8, 0.8, 0.7, 0.7, 0.6, 0.5, 1.0)
num_labels = 3755

# reformat data to the intended dim
def reformat(dataset, labels):
    dataset = dataset.reshape(
        (-1, num_channels, image_height, image_width)).transpose(0,2,3,1).astype(np.float32)
    dataset = dataset / 255
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.uint32).reshape((-1, 3755))
    return dataset, labels

# shuffle the data and label accordingly
def shuffle(dataset, labels):
    perm = np.random.permutation(len(dataset))
    dataset = dataset[perm]
    labels = labels[perm]
    return dataset, labels

# calculate training or testing accuracy
def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
        / predictions.shape[0])

# Data prepare
if(args["test_mode"] <= 0):
    print("[INFO] using training mode")
    print("[INFO] loading features...")
    data_file = tables.open_file(data_path, mode='r')
    print("[INFO] loading labels...")
    label_file = tables.open_file(label_path, mode='r')
    testData = data_file.root.trainData[918970:919970]
    testLabels = label_file.root.trainLabel[918970:919970]
    testData, testLabels = reformat(testData, testLabels)
#if(args["test_mode"] <= 0):
#    print("[INFO] using training mode")
#    print("[INFO] loading features...")
#    features = open(data_path)
#    totalData = features.readline().strip('\t').split('\t')
#    totalData = np.asarray(totalData, dtype='float32')
#    print("[INFO] finished loading features from %s" %data_path)
#    print("[INFO] loading labels...")
#    labels = open(label_path)
#    totalLabels = labels.readline().strip('\t').split('\t')
#    totalLabels = np.asarray(totalLabels, dtype='int32')
#    print("[INFO] finished loading labels from %s" %label_path)
#    totalData, totalLabels = reformat(totalData, totalLabels)
#    # restrict data to [0, 1]
#    totalData = totalData / 255
#    train_size =(int)(0.9 * totalData.shape[0])
#    train_index = np.random.choice(totalData.shape[0], train_size, replace=False)
#    test_index = np.asarray(list(set(np.arange(totalData.shape[0])) - set(train_index)))
#    trainData = totalData[train_index]
#    testData = totalData[test_index]
#    trainLabels = totalLabels[train_index]
#    testLabels = totalLabels[test_index]
#
#    print('[INFO] Training set', trainData.shape, trainLabels.shape)
#    print('[INFO] Test set', testData.shape, testLabels.shape)
#
else:
    print("[INFO] using single image test mode!")
    #testData = np.array(Image.open(args["image_path"]), dtype='float32')
    #testData = np.reshape(testData, (1, image_height, image_width, num_channels))
    
# constructing stage
graph = tf.Graph()

with graph.as_default():

    # Input data.
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_height, image_width, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

    tf_test_dataset = tf.constant(testData)

    # Eight CONV layer variables, in truncated normal distribution.
    conv_weights = []
    conv_biases = []
    conv_weights.append(tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, num_channels, depth[0]], stddev=0.1) / (patch_size * patch_size * num_channels)))
    conv_biases.append(tf.Variable(tf.zeros(depth[0])))
    for i in range(1, 8):
        conv_weights.append(tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, depth[i-1], depth[i]], stddev=0.1) / (patch_size * patch_size * depth[i-1])))
        conv_biases.append(tf.Variable(tf.zeros(depth[i])))

    # Three FC layer variables
    fc_weights = []
    fc_biases = []
    fc_weights.append(tf.Variable(tf.truncated_normal(
      [image_height // 16 * image_width // 16 * depth[7], num_hidden[0]], stddev=0.1) / image_height // 16 * image_width // 16 * depth[7]))
    fc_biases.append(tf.Variable(tf.zeros(num_hidden[0])))
    
    fc_weights.append(tf.Variable(tf.truncated_normal(
      [num_hidden[0], num_hidden[1]], stddev=0.1) / num_hidden[0]))
    fc_biases.append(tf.Variable(tf.zeros(num_hidden[1])))

    softmax_weights = tf.Variable(tf.truncated_normal(
      [num_hidden[1], num_labels], stddev=0.1) / num_hidden[1])
    softmax_biases = tf.Variable(tf.zeros(num_labels))    

    # Model.
    def model(data):
        # conv layers
        conv1 = tf.nn.conv2d(data, conv_weights[0], [1, 1, 1, 1], padding='SAME')
        hidden1 = tf.nn.relu(conv1 + conv_biases[0])
        hidden1 = tf.nn.dropout(hidden1, keep_prob[0])
        conv2 = tf.nn.conv2d(hidden1, conv_weights[1], [1, 1, 1, 1], padding='SAME')
        hidden2 = tf.nn.relu(conv2 + conv_biases[1])
        hidden2 = tf.nn.dropout(hidden2, keep_prob[1])
        pool1 = tf.nn.max_pool(hidden2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                             padding='SAME')
        
        conv3 = tf.nn.conv2d(pool1, conv_weights[2], [1, 1, 1, 1], padding='SAME')
        hidden3 = tf.nn.relu(conv3 + conv_biases[2])
        hidden3 = tf.nn.dropout(hidden3, keep_prob[2])
        conv4 = tf.nn.conv2d(hidden3, conv_weights[3], [1, 1, 1, 1], padding='SAME')
        hidden4 = tf.nn.relu(conv4 + conv_biases[3])
        hidden4 = tf.nn.dropout(hidden4, keep_prob[3])
        pool2 = tf.nn.max_pool(hidden4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                             padding='SAME')
        
        conv5 = tf.nn.conv2d(pool2, conv_weights[4], [1, 1, 1, 1], padding='SAME')
        hidden5 = tf.nn.relu(conv5 + conv_biases[4])
        hidden5 = tf.nn.dropout(hidden5, keep_prob[4])
        conv6 = tf.nn.conv2d(hidden5, conv_weights[5], [1, 1, 1, 1], padding='SAME')
        hidden6 = tf.nn.relu(conv6 + conv_biases[5])
        hidden6 = tf.nn.dropout(hidden6, keep_prob[5])
        pool3 = tf.nn.max_pool(hidden6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                             padding='SAME')
        
        conv7 = tf.nn.conv2d(pool3, conv_weights[6], [1, 1, 1, 1], padding='SAME')
        hidden7 = tf.nn.relu(conv7 + conv_biases[6])
        hidden7 = tf.nn.dropout(hidden7, keep_prob[6])
        conv8 = tf.nn.conv2d(hidden7, conv_weights[7], [1, 1, 1, 1], padding='SAME')
        hidden8 = tf.nn.relu(conv8 + conv_biases[7])
        hidden8 = tf.nn.dropout(hidden8, keep_prob[7])
        pool4 = tf.nn.max_pool(hidden8, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                             padding='SAME')
        # Optional: normalization
        # norm1 = tf.nn.lrn(pool1, 4, bias=1.0,   alpha=0.001 / 9.0, beta=0.75,
        #                 name='norm1')

        # fc layers
        shape = pool4.get_shape().as_list()
        reshape = tf.reshape(pool4, [shape[0], shape[1]*shape[2]*shape[3]])
        hidden9 = tf.nn.relu(tf.matmul(reshape, fc_weights[0]) + fc_biases[0])
        hidden9 = tf.nn.dropout(hidden9, keep_prob[8])
        
        hidden10 = tf.nn.relu(tf.matmul(hidden9, fc_weights[1]) + fc_biases[1])
        hidden10 = tf.nn.dropout(hidden10, keep_prob[9])
        
        # output layer
        result = tf.matmul(hidden10, softmax_weights) + softmax_biases

        return result

    if(args["test_mode"] <= 0):
        # Training loss and pred computation.
        logits = model(tf_train_dataset)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
        train_prediction = tf.nn.softmax(logits)
        test_prediction = tf.nn.softmax(model(tf_test_dataset))
        # Learning rate decay
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 5e-3
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               100000, 0.96, staircase=True)
        
        momentum = 0.9
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(loss)
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
    else:
        test_prediction = tf.nn.softmax(model(tf_test_dataset))

# running stage
num_epochs = 10
num_iters = 910

with tf.Session(graph=graph) as session:
    begin = clock()
    if(args["load_model"] > 0):
        print('[INFO] restoring model from file...')
        saver.restore(session, args['model_path'])
        print('[INFO] model restored.')
    else:
        print('[INFO] initializing model from scratch...')
        tf.initialize_all_variables().run()
        print('[INFO] model Initialized.')
    if(args["test_mode"] <= 0):
        for epoch in range(num_epochs):
            #trainData, trainLabels = shuffle(trainData, trainLabels)
            trainIndex = np.random.permutation(919975)
            offset = 0
            for iteration in range(num_iters):
                # stochastic gradient descent
                #batch_index = np.random.choice(trainLabels.shape[0], batch_size)
                #batch_data = trainData[batch_index]
                #batch_labels = trainLabels[batch_index]
                # batch gradient descent
                offset = (iteration * batch_size)
                if(offset + batch_size > 919975):
                    offset = 0
                batch_data = np.zeros((batch_size, 8192))
                batch_labels = np.zeros((batch_size, 1))
                for i in range(batch_size):
                    batch_data[i] = data_file.root.trainData[trainIndex[offset+i]]
                    batch_labels[i] = label_file.root.trainLabel[trainIndex[offset+i]]
                #batch_data = data_file.root.trainData[trainIndex[offset:(offset + batch_size)]]
                #batch_labels = label_file.root.trainLabel[trainIndex[offset:(offset + batch_size)]]
                batch_data, batch_labels = reformat(batch_data, batch_labels)
                feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
            
                _, l, predictions = session.run(
                    [optimizer, loss, train_prediction], feed_dict=feed_dict)
                #if(iteration == 100):
                    #np.set_printoptions(threshold=np.nan)
                    #print('[TEST] Softmax weights:')
                    #print(tf.Print(softmax_weights))
                    #print('[TEST] Batch predictions:')
                    #print(predictions[0])
                    #print('[TEST] Real labels:')
                    #print(batch_labels[0])
                if (iteration % 100 == 0):
                    np.set_printoptions(threshold=np.nan)
                    print('[TEST] Softmax weights:')
                    print(tf.Print(softmax_weights, [softmax_weights]))
                    print('[INFO] Minibatch loss at epoch %d iteration %d: %f' % (epoch, iteration, l))
                    print('[INFO] Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
                    print('[INFO] Test accuracy: %.1f%%' % accuracy(test_prediction.eval(session=session), testLabels))
    else:
        print('[INFO] test prediction: mostlikely to be %s' %np.argmax(test_prediction.eval(session=session)))
    if(args["save_model"] > 0):
        print('[INFO] saving model to file...')
        save_path = saver.save(session, args["model_path"]) 
        print("[INFO] Model saved in file: %s" % save_path)
    else:
        print('[INFO] you chose not to save model and exit.')
end = clock()
print('[INFO] total time used: %f' %(end - begin))
