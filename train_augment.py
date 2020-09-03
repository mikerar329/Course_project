import math
import random
import argparse
import glob
import numpy as np
import tensorflow as tf
import time
from tensorflow.python import keras as keras


LOG_DIR = 'logs'
SHUFFLE_BUFFER = 10
BATCH_SIZE = 8
NUM_CLASSES = 2
PARALLEL_CALLS = 4
RESIZE_TO = 224
TRAINSET_SIZE = 1602
VALSET_SIZE = 586


def parse_proto_example(proto):
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/class/label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64))
    }
    example = tf.parse_single_example(proto, keys_to_features)
    example['image'] = tf.image.decode_jpeg(example['image/encoded'], channels=3)
    example['image'] = tf.image.convert_image_dtype(example['image'], dtype=tf.float32)
    example['image'] = tf.image.resize_images(example['image'], tf.constant([RESIZE_TO, RESIZE_TO]))
    return example['image'], example['image/class/label']


def normalize(image, label):
    return tf.image.per_image_standardization(image), label

def resize(image, label):
    return tf.image.resize_images(image, tf.constant([RESIZE_TO, RESIZE_TO])), label

def create_dataset(filenames, batch_size):

    return tf.data.TFRecordDataset(filenames)\
        .map(parse_proto_example)\
        .map(resize)\
        .map(normalize)\
        .shuffle(buffer_size=5 * batch_size)\
        .repeat()\
        .batch(batch_size)\
        .prefetch(2 * batch_size)

def create_augmented_dataset(filenames, batch_size):

    return tf.data.TFRecordDataset(filenames)\
        .map(parse_proto_example) \
        .map(resize) \
        .map(normalize)\
        .map(augmented_train) \
        .map(resize) \
        .shuffle(buffer_size=5 * batch_size)\
        .repeat()\
        .batch(batch_size)\
        .prefetch(2 * batch_size)

def augmented_train(image, label):
    # image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.random_crop(image, size=[200, 200, 3], seed=None, name=None)
    image = tf.image.random_flip_left_right(image)
    degree = 15
    dgr = random.uniform(-degree, degree)
    image = tf.contrib.image.rotate(image, dgr * math.pi / 180, interpolation='BILINEAR')
    return image, label

class Validation(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, validation_files, batch_size):
        self.log_dir = log_dir
        self.validation_files = validation_files
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        print(' The average loss for epoch {} is {:7.2f} '.format(
            epoch, logs['loss']
        ))

        validation_dataset = create_dataset(self.validation_files, self.batch_size)
        validation_images, validation_labels = validation_dataset.make_one_shot_iterator().get_next()
        validation_labels = tf.one_hot(validation_labels, NUM_CLASSES)

        result = self.model.evaluate(
            validation_images,
            validation_labels,
            steps=int(np.ceil(VALSET_SIZE / float(BATCH_SIZE)))
        )
        callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, update_freq='epoch', batch_size=self.batch_size)

        callback.set_model(self.model)
        callback.on_epoch_end(epoch, {
            'val_' + self.model.metrics_names[i]: v for i, v in enumerate(result)
        })
 

def build_model():
    model = keras.models.load_model('model3.h5')
    model.trainable = True
    return model
     

def main():
    args = argparse.ArgumentParser()
    args.add_argument('--train', type=str, help='Glob pattern to collect train tfrecord files')
    args.add_argument('--test', type=str, help='Glob pattern to collect test tfrecord files')
    args = args.parse_args()
    print("augment0_lr100")
    train_dataset = create_augmented_dataset(glob.glob(args.train), BATCH_SIZE)
    train_images, train_labels = train_dataset.make_one_shot_iterator().get_next()
    train_labels = tf.one_hot(train_labels, NUM_CLASSES)

    model = build_model()
    #  optimizer=keras.optimizers.SGD(lr=0.0001, momentum=0.9, nesterov=True),
    # поставить RMSprop(learning_rate=0.001)
    model.compile(
        optimizer=keras.optimizers.rmsprop(lr=0.00001),
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=[tf.keras.metrics.categorical_accuracy],
        target_tensors=[train_labels]
    )

    log_dir='{}\logs\logs_augment-{}'.format(LOG_DIR, time.time())
    model.fit(
        (train_images, train_labels),
        epochs=30,
        steps_per_epoch=int(np.ceil(TRAINSET_SIZE / float(BATCH_SIZE))),
        callbacks=[
            tf.keras.callbacks.TensorBoard(log_dir),
            Validation(log_dir, validation_files=glob.glob(args.test), batch_size=BATCH_SIZE)
        ]
    )


if __name__ == '__main__':
    main()


