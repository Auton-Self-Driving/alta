import glob
import os
import tensorflow as tf
import numpy as np

import vis_module
import ae.util as util
import ae.plot_cm as plot_cm
from ae.controller import AEController

import tensorboard_logging as tf_log


frame_stack = None
fmt = None


def prepare_for_training(dataset, args, cache=True, shuffle_buffer_size=1000):
    if cache:
        if isinstance(cache, str):
            dataset = dataset.cache(cache)
        else:
            dataset = dataset.cache()

    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    dataset = dataset.batch(args.batch_size)
    # dataset = dataset.repeat()
    dataset = dataset.prefetch(buffer_size=32)

    return dataset

def convert_if_not_one_hot(arr):
    if arr.ndim == 2:
        return (np.arange(arr.max() + 1) == arr[...,None]).astype(int)

def read_npy_file(paths):
    final_data = []
    for path in paths:
        path = path.decode()
        tokens = path.split('/')
        parent_dir = os.path.join('/', *tokens[:-1])
        f = tokens[-1]
        index = int(f.split(fmt)[0])
        data = []
        for fs in range(frame_stack):
            path = os.path.join(parent_dir, "{:08d}{}".format(index + fs, fmt))
            if os.path.exists(path):
                if fmt == '.npy':
                    data.append(convert_if_not_one_hot(np.load(path)))
                else:
                    data.append(convert_if_not_one_hot(np.load(path)['img']))
            else:
                path = os.path.join(parent_dir, "{:08d}{}".format(index, fmt))
                if fmt == '.npy':
                    data.append(convert_if_not_one_hot(np.load(path)))
                else:
                    data.append(convert_if_not_one_hot(np.load(path)['img']))
        data = np.concatenate(data, axis=2)
        final_data.append(data)
    final_data = np.array(final_data).astype(np.float32)
    return final_data


def get_scratch_dir(base_log_dir):
    return base_log_dir.split(base_log_dir.split("/home")[0])[1].replace("/home", "/home/scratch")

def write_inds_to_file(filename, image_paths):
    with open("ae/{}.txt".format(filename), "w") as f:
        for path in image_paths:
            f.write(path + "\n")
            
def read_inds_from_file(filename):
    with open("ae/{}.txt".format(filename), "r") as f:
        inds = [ind.strip() for ind in f.readlines()]
    return inds

def split_train_test_data(args):
    image_inds = np.array([f.split(args.fmt)[0] for f in os.listdir(args.data_dir) if os.path.isfile(os.path.join(args.data_dir, f))])
    import pdb; pdb.set_trace()
    write_inds_to_file('trainval', image_inds)
    
    total_elements = len(image_inds)
    test_elements = int(total_elements / 10)
    
    test_inds = np.random.choice(total_elements, test_elements, replace=False)
    train_inds = np.delete(np.arange(total_elements), test_inds)

    test_image_inds = image_inds[test_inds]
    write_inds_to_file('val', test_image_inds)
    test_image_paths = np.array([os.path.join(args.data_dir, "{}{}".format(f, args.fmt)) for f in test_image_inds])
    
    train_image_inds = image_inds[train_inds]
    write_inds_to_file('train', train_image_inds)
    train_image_paths = np.array([os.path.join(args.data_dir, "{}{}".format(f, args.fmt)) for f in train_image_inds])
    
    return train_image_paths, test_image_paths

def load_train_test_data(args):
    train_image_inds = read_inds_from_file('train')
    train_image_paths = np.array([os.path.join(args.data_dir, "{}{}".format(f, args.fmt)) for f in train_image_inds])

    test_image_inds = read_inds_from_file('val')
    test_image_paths = np.array([os.path.join(args.data_dir, "{}{}".format(f, args.fmt)) for f in test_image_inds])
    
    return train_image_paths, test_image_paths

def train_vae(args, prefix, config):
    
    if args.new_data_split:
        # Generate Train & Test image paths
        train_image_paths, test_image_paths = split_train_test_data(args)
    else:
        # Load Train & Test image paths
        train_image_paths, test_image_paths = load_train_test_data(args)

    global frame_stack, fmt
    
    if args.frame_stack:
        frame_stack = args.frame_stack
    
    fmt = args.fmt

    ALTA_LOGS = args.base_log_dir + prefix
    if not os.path.exists(ALTA_LOGS):
        os.makedirs(ALTA_LOGS)

    if os.path.exists('/home/scratch'):
        SCRATCH_DIR = os.path.join(get_scratch_dir(args.base_log_dir), prefix.split('_runid_')[0], prefix)
    else:
        SCRATCH_DIR = ALTA_LOGS
    
    IMAGES_PATH = os.path.join(SCRATCH_DIR, 'images')
    VIDEO_PATH = None
    IMAGES_PATH_VAE = os.path.join(SCRATCH_DIR, 'images_VAE')
    VIDEO_PATH_VAE = None
    MODEL_DIR = os.path.join(SCRATCH_DIR, 'ae_weights')
    CM_PATH = os.path.join(SCRATCH_DIR, 'CM_images')
    CM_PATH_TEST = os.path.join(SCRATCH_DIR, 'CM_images_test')
    
    TB_LOGS_DIR = os.path.join(ALTA_LOGS, 'tb')
    logger = tf_log.Logger(TB_LOGS_DIR)
    
    VIDEO_FRAME_SKIP = 1
    NUM_CLASSES = 5

    vis_wrapper = vis_module.vis(IMAGES_PATH, VIDEO_PATH, frame_skip=VIDEO_FRAME_SKIP)
    vis_wrapper_vae = vis_module.vis(IMAGES_PATH_VAE, VIDEO_PATH_VAE, frame_skip=VIDEO_FRAME_SKIP)

    ae_controller = AEController(image_size=(128, 128, NUM_CLASSES * args.frame_stack), learning_rate=args.lr,
                batch_size=args.batch_size, buffer_size=1, frame_stack=args.frame_stack)

    if args.model_path:
        print('Loading pretrained VAE!')
        ae_controller.load(args.model_path)

    sess = ae_controller.ae.sess

    with ae_controller.ae.g.as_default():
        
        # Create train dataset
        train_paths_placeholder = tf.placeholder(train_image_paths.dtype, train_image_paths.shape)
        train_dataset = tf.data.Dataset.from_tensor_slices((train_paths_placeholder))
        train_dataset = prepare_for_training(train_dataset, args, cache=True, shuffle_buffer_size=1000)
        train_dataset = train_dataset.map(lambda item: tuple(tf.py_func(read_npy_file, [item], [tf.float32,])), num_parallel_calls=16)
        
        train_iterator = train_dataset.make_initializable_iterator()
        train_next_element = train_iterator.get_next()
        
        # Create test dataset
        test_paths_placeholder = tf.placeholder(test_image_paths.dtype, test_image_paths.shape)
        test_dataset = tf.data.Dataset.from_tensor_slices((test_paths_placeholder))
        test_dataset = prepare_for_training(test_dataset, args, cache=True, shuffle_buffer_size=1000)
        test_dataset = test_dataset.map(lambda item: tuple(tf.py_func(read_npy_file, [item], [tf.float32,])), num_parallel_calls=16)
        
        test_iterator = test_dataset.make_initializable_iterator()
        test_next_element = test_iterator.get_next()

        train_step = 0
        train_epoch_losses = []
        train_epoch_accuracies = []
        train_epoch_my_accuracies = []
        train_epoch_cms = []
        
        test_epoch_losses = []
        test_epoch_accuracies = []
        test_epoch_my_accuracies = []
        test_epoch_cms = []

        for epoch in range(args.epochs):
            
            train_batch_losses = []
            train_batch_accuracies = []
            train_batch_my_accuracies = []
            train_epoch_cm = 0

            sess.run(train_iterator.initializer,
                    feed_dict={train_paths_placeholder: train_image_paths})

            try:
                while True:
                    batch = sess.run(train_next_element)
                    batch = np.reshape(batch[0], (-1, 128, 128, NUM_CLASSES * frame_stack))
                    feed = {ae_controller.ae.x: batch, }
                    (train_loss, train_step, accuracy, accuracy_op, my_accuracy, confusion_matrix, _) = sess.run([
                        ae_controller.ae.loss,
                        ae_controller.ae.global_step,
                        ae_controller.ae.accuracy,
                        ae_controller.ae.accuracy_op,
                        ae_controller.ae.my_accuracy,
                        ae_controller.ae.confusion_matrix,
                        ae_controller.ae.train_op
                    ], feed)

                    if ((train_step + 1) % 50 == 0):
                        print("AE: Epoch:{}/{}, global_step:{}, loss={}, my_accuracy={}, accuracy={}, accuracy_op={}".format(
                                epoch + 1, args.epochs, train_step + 1, train_loss, my_accuracy, accuracy, accuracy_op))
                        if not os.path.exists(MODEL_DIR):
                            os.makedirs(MODEL_DIR)
                        ae_controller.save(os.path.join(MODEL_DIR, 'ae_{}'.format(epoch + 1)))

                    train_batch_losses.append(train_loss)
                    train_batch_accuracies.append(accuracy)
                    train_batch_my_accuracies.append(my_accuracy)
                    train_epoch_cm += confusion_matrix
                    
                    logger.log_scalar('train/batch_losses', train_loss, train_step)
                    logger.log_scalar('train/batch_accuracies', accuracy, train_step)
                    logger.log_scalar('train/batch_my_accuracies', my_accuracy, train_step)

            except tf.errors.OutOfRangeError:
                epoch_loss = np.mean(np.array(train_batch_losses))
                epoch_accuracy = np.mean(np.array(train_batch_accuracies))
                epoch_my_accuracy = np.mean(np.array(train_batch_my_accuracies))
                eps = 1e-8
                normalization = np.sum(train_epoch_cm, axis=1).reshape((-1, 1)) + eps
                train_epoch_cm_normalized =  train_epoch_cm / normalization
                
                train_epoch_losses.append(epoch_loss)
                train_epoch_accuracies.append(epoch_accuracy)
                train_epoch_my_accuracies.append(epoch_my_accuracy)
                train_epoch_cms.append(train_epoch_cm_normalized)

                logger.log_scalar('train/epoch_losses', epoch_loss, epoch + 1) 
                logger.log_scalar('train/epoch_accuracies', epoch_accuracy, epoch + 1)
                logger.log_scalar('train/epoch_my_accuracies', epoch_my_accuracy, epoch + 1)
                plot_cm.save_cm(train_epoch_cm_normalized, CM_PATH , epoch + 1)
            
            ae_controller.set_target_params()
            
            if epoch % 1 == 0:
                test_batch_losses = []
                test_batch_accuracies = []
                test_batch_my_accuracies = []
                test_epoch_cm = 0

                sess.run(test_iterator.initializer,
                        feed_dict={test_paths_placeholder: test_image_paths})

                try:
                    while True:
                        batch = sess.run(test_next_element)
                        batch = np.reshape(batch[0], (-1, 128, 128, NUM_CLASSES * frame_stack))
                        feed = {ae_controller.ae.x: batch, }
                        ind = np.random.randint(0, batch.shape[0])
                        
                        (test_loss, accuracy, accuracy_op, my_accuracy, confusion_matrix, preds) = sess.run([
                            ae_controller.ae.loss,
                            ae_controller.ae.accuracy,
                            ae_controller.ae.accuracy_op,
                            ae_controller.ae.my_accuracy,
                            ae_controller.ae.confusion_matrix,
                            ae_controller.ae.output_preds
                        ], feed)

                        test_batch_losses.append(test_loss)
                        test_batch_accuracies.append(accuracy)
                        test_batch_my_accuracies.append(my_accuracy)
                        test_epoch_cm += confusion_matrix
                        
                        label = np.reshape(batch[ind], (128, -1, NUM_CLASSES))
                        label = util.convert_from_one_hot(np.hstack((label[:, x::frame_stack, :] for x in range(frame_stack))))
                        preds = np.reshape(preds, (args.batch_size, 128, -1, frame_stack))[ind]
                        preds = np.hstack((preds[:, :, x] for x in range(frame_stack)))
                        
                        label = util.convert_to_rgb(label, reduced_classes=True)
                        preds = util.convert_to_rgb(preds, reduced_classes=True)

                        vis_wrapper.save_image(label, 1)
                        vis_wrapper_vae.save_image(preds, 1)
  
                except tf.errors.OutOfRangeError:
                    epoch_loss = np.mean(np.array(test_batch_losses))
                    epoch_accuracy = np.mean(np.array(test_batch_accuracies))
                    epoch_my_accuracy = np.mean(np.array(test_batch_my_accuracies))
                    eps = 1e-8
                    normalization = np.sum(test_epoch_cm, axis=1).reshape((-1, 1)) + eps
                    test_epoch_cm_normalized =  test_epoch_cm / normalization
                    
                    test_epoch_losses.append(epoch_loss)
                    test_epoch_accuracies.append(epoch_accuracy)
                    test_epoch_my_accuracies.append(epoch_my_accuracy)
                    test_epoch_cms.append(test_epoch_cm_normalized)

                    logger.log_scalar('test/epoch_losses', epoch_loss, epoch + 1)
                    logger.log_scalar('test/epoch_accuracies', epoch_accuracy, epoch + 1)
                    logger.log_scalar('test/epoch_my_accuracies', epoch_my_accuracy, epoch + 1)
                    plot_cm.save_cm(test_epoch_cm_normalized, CM_PATH_TEST , epoch + 1)

        train_epoch_losses = np.array(train_epoch_losses)
        train_epoch_accuracies = np.array(train_epoch_accuracies)
        train_epoch_my_accuracies = np.array(train_epoch_my_accuracies)
        train_epoch_cms = np.array(train_epoch_cms)
        
        test_epoch_losses = np.array(test_epoch_losses)
        test_epoch_accuracies = np.array(test_epoch_accuracies)
        test_epoch_my_accuracies = np.array(test_epoch_my_accuracies)
        test_epoch_cms = np.array(test_epoch_cms)

        return train_epoch_losses, train_epoch_accuracies, train_epoch_my_accuracies, train_epoch_cms, test_epoch_losses, test_epoch_accuracies, test_epoch_my_accuracies, test_epoch_cms

