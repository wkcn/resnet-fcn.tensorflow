import tensorflow as tf
def resave_tensors(file_name, to_name, rename_map, dry_run=False):
    """
    Updates checkpoint by renaming tensors in it.
    :param file_name: Filename with checkpoint.
    :param rename_map: Map from old names to new ones
    :param dry_run: If True, just print new tensors.
    """
    renames_count = 0
    reader = tf.train.NewCheckpointReader(file_name)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in sorted(var_to_shape_map):
        tensor_val = reader.get_tensor(key)
        #print('shape: {}'.format(tensor_val.shape))
        r = False
        if key in rename_map:
            renames_count += 1
            key = rename_map[key]
            r = True
        print("tensor_name: ", key, r)
        tf.Variable(tensor_val, dtype=tensor_val.dtype, name=key)
    saver = tf.train.Saver()
    if not dry_run:
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            saver.save(session, to_name)
    print('Renamed vars: {}'.format(renames_count))

if __name__ == "__main__":
    PATH = "/home/wkcn/Downloads/tensorflow-resnet-pretrained-20160509/"
    CHECKPOINT_FN = PATH + "ResNet-L50.ckpt"
