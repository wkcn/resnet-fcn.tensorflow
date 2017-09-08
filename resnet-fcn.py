import tensorflow as tf
import numpy as np
import resnet

PATH = "./"
META_FN = PATH + "ResNet-L50.meta"
CHECKPOINT_FN = PATH + "ResNet-L50.ckpt"

def get_deconv_filter(f_shape):
    width = f_shape[0]
    heigh = f_shape[0]
    f = np.ceil(width/2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([f_shape[0], f_shape[1]])
    for x in range(width):
        for y in range(heigh):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weights = np.zeros(f_shape)
    for i in range(f_shape[2]):
        weights[:, :, i, i] = bilinear

    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)
    var = tf.get_variable(name="up_filter", initializer=init,
                          shape=weights.shape)
    return var

def upscore_layer(x, shape, num_classes, name, ksize, stride):
    strides = [1, stride, stride, 1]
    with tf.variable_scope(name):
        in_features = x.get_shape()[3].value
        if shape is None:
            in_shape = tf.shape(x)
            h = ((in_shape[1] - 1) * stride) + 1
            w = ((in_shape[2] - 1) * stride) + 1
            new_shape = [in_shape[0], h, w, num_classes]
        else:
            new_shape = [shape[0], shape[1], shape[2], num_classes]
        output_shape = tf.stack(new_shape)
        f_shape = [ksize, ksize, num_classes, in_features]
        num_input = ksize * ksize * in_features / stride
        stddev = (2 / num_input)**0.5
        weights = get_deconv_filter(f_shape)
        deconv = tf.nn.conv2d_transpose(x, weights, output_shape, strides = strides, padding='SAME')
        return deconv

def score_layer(x, name, num_classes, stddev = 0.001): 
    with tf.variable_scope(name) as scope:
        # get number of input channels
        in_features = x.get_shape()[3].value
        shape = [1, 1, in_features, num_classes]
        w_decay = 5e-4
        init = tf.truncated_normal_initializer(stddev = stddev)
        weights = tf.get_variable("weights", shape = shape, initializer = init)
        collection_name = tf.GraphKeys.REGULARIZATION_LOSSES

        if not tf.get_variable_scope().reuse:
            weight_decay = tf.multiply(tf.nn.l2_loss(weights), w_decay, name='weight_loss')
            tf.add_to_collection(collection_name, weight_decay)

        conv = tf.nn.conv2d(x, weights, [1, 1, 1, 1], padding='SAME')
        # Apply bias
        initializer = tf.constant_initializer(0.0)
        conv_biases = tf.get_variable(name='biases', shape=[num_classes],initializer=initializer)

        bias = tf.nn.bias_add(conv, conv_biases)

        return bias




def inference(sess, x, is_training,
              num_classes=1000,
              num_blocks=[3, 4, 6, 3],  # defaults to 50-layer network
              use_bias=False, # defaults to using batch norm
              bottleneck=True, pretrained_resnet = False):

    logits = resnet.inference(x, is_training, 1000, num_blocks, use_bias, bottleneck) 

    lst = [v for v in tf.global_variables()]

    if pretrained_resnet:
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess, CHECKPOINT_FN)
        print ("Load Pre-Model OK")

    scale2 = sess.graph.get_tensor_by_name("scale2/block3/Relu:0") # 56x56
    scale3 = sess.graph.get_tensor_by_name("scale3/block3/Relu:0") # 28x28
    scale4 = sess.graph.get_tensor_by_name("scale4/block3/Relu:0") # 14x14
    scale5 = sess.graph.get_tensor_by_name("scale5/block3/Relu:0") # 7x7

    with tf.variable_scope('scale_fcn'):
        upscore2 = upscore_layer(scale5, shape = tf.shape(scale4), num_classes = num_classes, name = "upscore2", ksize = 4, stride = 2) 
        score_scale4 = score_layer(scale4, "score_scale4", num_classes = num_classes)
        fuse_scale4 = tf.add(upscore2, score_scale4)

        upscore4 = upscore_layer(fuse_scale4, shape = tf.shape(scale3), num_classes = num_classes, name = "upscore4", ksize = 4, stride = 2) 
        score_scale3 = score_layer(scale3, "score_scale3", num_classes = num_classes)
        fuse_scale3 = tf.add(upscore4, score_scale3)

        upscore8 = upscore_layer(fuse_scale3, shape = tf.shape(scale2), num_classes = num_classes, name = "upscore8", ksize = 4, stride = 2) 
        score_scale2 = score_layer(scale2, "score_scale2", num_classes = num_classes)
        fuse_scale2 = tf.add(upscore8, score_scale2)

        upscore32 = upscore_layer(fuse_scale2, shape = tf.shape(x), num_classes = num_classes, name = "upscore32", ksize = 8, stride = 4)

        pred_up = tf.argmax(upscore32, dimension = 3)
        pred = tf.expand_dims(pred_up, dim = 3)
    print ("Build OK")
    return pred, upscore32


if __name__ == "__main__":
    sess = tf.Session()
    batch_size = 16
    num_classes = 5
    x = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
    labels = tf.placeholder(tf.int32, [batch_size, 224, 224])
    pred, logits = inference(sess, x, is_training = True, num_classes = num_classes, pretrained_resnet = True)
    sess.run(tf.global_variables_initializer())

    train_X = np.random.randint(0, 256, size = [batch_size, 224, 224, 3]).astype(np.float)
    train_y = np.random.randint(0, num_classes, [batch_size, 224, 224]).astype(np.int)

    print (sess.run(logits, feed_dict = {x : train_X, labels : train_y})) 
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = labels, name = "entropy")))
