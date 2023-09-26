import tensorflow as tf
from data_loader import get_tf_datasets
from models import StampClassifier16
from models import StampClassifierMultiScale, StampClassifierFull
from hp_search import balanced_xentropy
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import os
import sys


os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
N_CLASSES = 6

dataset_name = sys.argv[1]
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]

# Best hyperparameters according to hp search
hyperparameters_per_dataset = {
    'cropped': {
        'layer_1': 112,
        'layer_2': 27,
        'layer_3': 92,
        'layer_4': 121,
        'layer_5': 133,
        'learning_rate': 0.00034627,
        'dropout_rate': 0.866799
    },
    'low_res': {
        'layer_1': 63,
        'layer_2': 31,
        'layer_3': 108,
        'layer_4': 14,
        'layer_5': 22,
        'learning_rate': 0.001125,
        'dropout_rate': 0.73065
    },
    'full': {
        'layer_1': 37,
        'layer_2': 60,
        'layer_3': 29,
        'layer_4': 26,
        'layer_5': 31,
        'learning_rate': 0.0007589,
        'dropout_rate': 0.71726
    },
    'multiscale': {
        'layer_1': 76,
        'layer_2': 18,
        'layer_3': 54,
        'layer_4': 28,
        'layer_5': 69,
        'learning_rate': 0.0007445,
        'dropout_rate': 0.8478
    }
}

hyperparameters = hyperparameters_per_dataset[dataset_name]

# This hp are always the best ones
hyperparameters['with_batchnorm'] = True
hyperparameters['first_kernel_size'] = 3
hyperparameters['batch_size'] = 256


class NoImprovementStopper:
    def __init__(self, num_steps: int):
        self.num_steps = num_steps
        self.historic_max = 0.0
        self.steps_without_improvement = 0

    def should_break(self, current_value):
        if current_value > self.historic_max:
            self.historic_max = current_value
            self.steps_without_improvement = 0
            return False
        else:
            self.steps_without_improvement += 1

        if self.steps_without_improvement >= self.num_steps:
            return True
        else:
            return False


def train_model_and_save(hyperparameters, save_name):
    layer_size_keys = sorted([k for k in hyperparameters.keys() if 'layer_' in k])
    layer_sizes = [hyperparameters[k] for k in layer_size_keys]
    batch_size = hyperparameters['batch_size']
    dropout_rate = hyperparameters['dropout_rate']
    with_batchnorm = hyperparameters['with_batchnorm']
    first_kernel_size = hyperparameters['first_kernel_size']
    learning_rate = hyperparameters['learning_rate']

    with tf.device('/cpu:0'):
        training_dataset, validation_dataset, test_dataset, label_encoder = get_tf_datasets(
            dataset_name=dataset_name, batch_size=batch_size)

    stamps_shape = list(test_dataset.as_numpy_iterator())[0][0].shape[1:]
    if dataset_name == 'cropped' or dataset_name == 'low_res':
        stamp_classifier = StampClassifier16(
            stamps_shape, layer_sizes, dropout_rate,
            with_batchnorm, first_kernel_size, n_classes=N_CLASSES)
    elif dataset_name == 'multiscale':
        stamp_classifier = StampClassifierMultiScale(
            stamps_shape, n_levels=4, layer_sizes=layer_sizes,
            dropout_rate=dropout_rate, with_batchnorm=with_batchnorm,
            first_kernel_size=first_kernel_size, n_classes=N_CLASSES)
    elif dataset_name == 'full':
        stamp_classifier = StampClassifierFull(
            stamps_shape, layer_sizes, dropout_rate,
            with_batchnorm, first_kernel_size, n_classes=N_CLASSES)
    else:
        raise ValueError('hey!')

    for x, pos, y in test_dataset:
        print(stamp_classifier((x[:10], pos[:10])))
        break

    for v in stamp_classifier.trainable_variables:
        print(v.name, v.shape, np.prod(v.shape))

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    @tf.function()
    def train_step(samples, positions, labels):
        with tf.GradientTape() as tape:
            predictions = stamp_classifier((samples, positions), training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, stamp_classifier.trainable_variables)
        optimizer.apply_gradients(zip(gradients, stamp_classifier.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    logdir = f'./logs/{save_name}'
    train_writer = tf.summary.create_file_writer(logdir + '/train')
    val_writer = tf.summary.create_file_writer(logdir + '/val')
    test_writer = tf.summary.create_file_writer(logdir + '/test')

    def val_test_step(dataset, iteration, file_writer):
        prediction_list = []
        label_list = []
        for samples, pos, labels in dataset:
            predictions = stamp_classifier((samples, pos))
            prediction_list.append(predictions)
            label_list.append(labels)

        labels = tf.concat(label_list, axis=0)
        predictions = tf.concat(prediction_list, axis=0)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions.numpy().argmax(axis=1), average='macro')
        xentropy_test = balanced_xentropy(labels, predictions)

        with file_writer.as_default():
            tf.summary.scalar('precision', precision, step=iteration)
            tf.summary.scalar('recall', recall, step=iteration)
            tf.summary.scalar('f1', f1, step=iteration)
            tf.summary.scalar('loss', xentropy_test, step=iteration)

        return f1

    log_frequency = 50
    stopper = NoImprovementStopper(5)

    for iteration, training_batch in enumerate(training_dataset):
        x_batch, pos_batch, y_batch = training_batch
        train_step(x_batch, pos_batch, y_batch)
        if iteration % log_frequency == 0 and iteration != 0:
            with train_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=iteration)
                tf.summary.scalar('accuracy', train_accuracy.result(), step=iteration)

        if iteration % 500 == 0:
            val_f1 = val_test_step(validation_dataset, iteration, val_writer)
            val_test_step(test_dataset, iteration, test_writer)
            if stopper.should_break(val_f1):
                break

        # Reset the metrics for the next iteration
        train_loss.reset_states()
        train_accuracy.reset_states()

    train_writer.flush()
    val_writer.flush()
    test_writer.flush()

    stamp_classifier.save(save_name)


for i in range(5):
    print(f'run {i}/5')
    train_model_and_save(hyperparameters, f'{dataset_name}_run_{i}')
