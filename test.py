import tensorflow as tf
import numpy as np
import resnet_fcn

if __name__ == "__main__":
    sess = tf.Session()
    batch_size = 16
    num_classes = 5

    x = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
    labels = tf.placeholder(tf.int32, [batch_size, 224, 224])

    pred, logits = resnet_fcn.inference(x, is_training = True, num_classes = num_classes)

    sess.run(tf.global_variables_initializer())

    train_X = np.random.randint(0, 256, size = [batch_size, 224, 224, 3]).astype(np.float)
    train_y = np.random.randint(0, num_classes, [batch_size, 224, 224]).astype(np.int)

    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = labels, name = "entropy")))
    print (sess.run(loss, feed_dict = {x : train_X, labels : train_y})) 
