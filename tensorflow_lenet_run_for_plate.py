import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
import argparse
from PIL import Image
from time import clock

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
    help="(optionall) path to the image if you are using test mode" )
args = vars(ap.parse_args())

data_path = "../data/matrices.txt"
label_path = "../data/classes.txt"

# declare some hyperparameters
batch_size = 500
patch_size = 5
depth = (32, 64)
num_hidden = 1024
image_height = 28
image_width = 18
num_labels = 65
num_channels = 1

def reformat(dataset, labels):
    dataset = dataset.reshape(
        (-1, image_height, image_width, num_channels)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
        / predictions.shape[0])

# Preprocessing
if(args["test_mode"] <= 0):
    print("[INFO] using training mode")
    print("[INFO] loading features...")
    features = open(data_path)
    print("[INFO] finished loading features from %s" %data_path)
    totalData = features.readline().strip('\t').split('\t')
    totalData = np.asarray(totalData, dtype='float32').reshape((-1, image_height, image_width))
    totalData = totalData / 255
    #split = (int)(0.9 * totalData.shape[0])
    #trainData = totalData[:split]
    #testData = totalData[split:]
    train_size =(int)(0.9 * totalData.shape[0])
    train_index = np.random.choice(totalData.shape[0], train_size, replace=False)
    test_index = np.asarray(list(set(np.arange(totalData.shape[0])) - set(train_index)))
    trainData = totalData[train_index]
    testData = totalData[test_index]

    print("[INFO] loading labels...")
    labels = open(label_path)
    print("[INFO] finished loading labels from %s" %label_path)
    totalLabels = labels.readline().strip('\t').split('\t')
    totalLabels = np.asarray(totalLabels, dtype='int32')
    trainLabels = totalLabels[train_index]
    testLabels = totalLabels[test_index]

    trainData, trainLabels = reformat(trainData, trainLabels)
    testData, testLabels = reformat(testData, testLabels)
    print('[INFO] Training set', trainData.shape, trainLabels.shape)
    print('[INFO] Test set', testData.shape, testLabels.shape)

else:
    print("[INFO] using single image test mode!")
    testData = np.array(Image.open(args["image_path"]), dtype='float32')
    testData = np.reshape(testData, (1, image_height, image_width, num_channels))
    
# constructing stage
graph = tf.Graph()

with graph.as_default():

    # Input data.
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_height, image_width, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

    tf_test_dataset = tf.constant(testData)

    # First CONV layer variables, in truncated normal distribution.
    layer1_weights = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, num_channels, depth[0]], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros(depth[0]))

    # dropout parameter
    keep_prob = tf.placeholder("float")

    # Second CONV layer variables
    layer2_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, depth[0], depth[1]], stddev=0.1))
    layer2_biases = tf.Variable(tf.zeros(depth[1]))

    # Three FC layer variables
    layer3_weights = tf.Variable(tf.truncated_normal(
      [image_height // 4 * image_width // 2 * depth[1], num_hidden], stddev=0.1))
    layer3_biases = tf.Variable(tf.zeros(num_hidden))

    layer4_weights = tf.Variable(tf.truncated_normal(
      [num_hidden, num_labels], stddev=0.1))
    layer4_biases = tf.Variable(tf.zeros(num_labels))    

    # Model.
    def model(data):
        # conv layer1
        conv1 = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
        hidden1 = tf.nn.relu(conv1 + layer1_biases)
        pool1 = tf.nn.max_pool(hidden1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                             padding='SAME', name='pool1')
        # Optional: normalization
        # norm1 = tf.nn.lrn(pool1, 4, bias=1.0,   alpha=0.001 / 9.0, beta=0.75,
        #                 name='norm1')   

        # conv layer2
        conv2 = tf.nn.conv2d(pool1, layer2_weights, [1, 1, 1, 1], padding='SAME')
        hidden2 = tf.nn.relu(conv2 + layer2_biases)
        pool2 = tf.nn.max_pool(hidden2, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1],
                             padding='SAME', name='pool2')
        # Optional: normalization
        # norm2 = tf.nn.lrn(pool2, 4, bias=1.0,   alpha=0.001 / 9.0, beta=0.75,
        #                 name='norm1')

        # FC layer1
        shape = pool2.get_shape().as_list()
        reshape = tf.reshape(pool2, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden3 = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)

        # Optional: dropout
        # hidden = tf.nn.dropout(hidden, keep_prob)

        # output layer
        result = tf.matmul(hidden3, layer4_weights) + layer4_biases

        return result

    # Training computation.
    logits = model(tf_train_dataset)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    test_prediction = tf.nn.softmax(model(tf_test_dataset))

    # Learning rate decay
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 1e-4
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           100000, 0.96, staircase=True)

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

# running stage
num_steps = 5000

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
        for step in range(num_steps):
            # stochastic gradient descent
            batch_index = np.random.choice(trainLabels.shape[0], batch_size)
            batch_data = trainData[batch_index]
            batch_labels = trainLabels[batch_index]
            # batch gradient descent
            #offset = (step * batch_size) % (trainLabels.shape[0] - batch_size)
            #batch_data = trainData[offset:(offset + batch_size), :, :, :]
            #batch_labels = trainLabels[offset:(offset + batch_size), :]
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob:0.5}
            
            _, l, predictions = session.run(
                [optimizer, loss, train_prediction], feed_dict=feed_dict)
            if (step % 100 == 0):
                print('[INFO] Minibatch loss at step %d: %f' % (step, l))
                print('[INFO] Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
                print('[INFO] Test accuracy: %.1f%%' % accuracy(test_prediction.eval(session=session,feed_dict={keep_prob:1.0}), testLabels))
    else:
        print('[INFO] test prediction: mostlikely to be %s' %np.argmax(test_prediction.eval(session=session,feed_dict={keep_prob:1.0})))
    if(args["save_model"] > 0):
        print('[INFO] saving model to file...')
        save_path = saver.save(session, args["model_path"]) 
        print("[INFO] Model saved in file: %s" % save_path)
    else:
        print('[INFO] you chose not to save model and exit.')
end = clock()
print('[INFO] total time used: %f' %(end - begin))
