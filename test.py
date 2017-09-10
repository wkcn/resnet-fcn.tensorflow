import tensorflow as tf
import numpy as np
import resnet_fcn
import time

PATH = "./"
META_FN = PATH + "ResNet-L50.meta"
CHECKPOINT_FN = PATH + "ResNet-L50.ckpt"


MOMENTUM = 0.9
UPDATE_OPS_COLLECTION = 'resnet_update_ops'  # must be grouped with training op

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', './logs',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_float('learning_rate', 0.01, "learning rate.")
tf.app.flags.DEFINE_integer('batch_size', 1, "batch size")
tf.app.flags.DEFINE_integer('max_steps', 500000, "max steps")



if __name__ == "__main__":
    sess = tf.Session()

    batch_size = FLAGS.batch_size
    num_classes = 5

    x = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
    labels = tf.placeholder(tf.int32, [batch_size, 224, 224])

    pred, logits = resnet_fcn.inference(x, is_training = True, num_classes = num_classes, num_blocks = [3,4,6,3])

    train_X = np.random.randint(0, 256, size = [batch_size, 224, 224, 3]).astype(np.float)
    train_y = np.random.randint(0, num_classes, [batch_size, 224, 224]).astype(np.int)

    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = labels, name = "entropy")))

    saver = tf.train.Saver([var for var in tf.global_variables() if "scale_fcn" not in var.name])

    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)

    opt = tf.train.MomentumOptimizer(FLAGS.learning_rate, MOMENTUM)
    grads = opt.compute_gradients(loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION)
    batchnorm_updates_op = tf.group(*batchnorm_updates)
    train_op = tf.group(apply_gradient_op, batchnorm_updates_op)


    tf.train.start_queue_runners(sess=sess)


    # Init global variables
    sess.run(tf.global_variables_initializer())

    # Restore variables
    restore_variables = True
    if restore_variables:
        saver.restore(sess, CHECKPOINT_FN)

    # new saver
    saver = tf.train.Saver(tf.global_variables())

    for _ in range(FLAGS.max_steps + 1):
        start_time = time.time()

        step = sess.run(global_step)

        run_op = [train_op, loss]

        o = sess.run(run_op, feed_dict = {
            x:train_X,
            labels:train_y
            })
        loss_value = o[1]

        duration = time.time() - start_time

        if step % 5 == 0:
            examples_per_sec = FLAGS.batch_size / float(duration)
            format_str = ('step %d, loss = %.2f (%.1f examples/sec; %.3f '
                          'sec/batch)')
            print(format_str % (step, loss_value, examples_per_sec, duration))

        if step > 1 and step % 500 == 0:
            checkpoint_path = os.path.join(FLAGS.train_dir, 'model%d.ckpt' % step)
            saver.save(sess, checkpoint_path, global_step=global_step)


    # print (sess.run(loss, feed_dict = {x : train_X, labels : train_y})) 

