import datetime
import fire
import os
import tensorflow as tf
import tensorflow_hub as hub

IMAGE_SIZE = [192, 192]


def read_tfrecord(example):

    features = {
        "image": tf.io.FixedLenFeature(
            [], tf.string
        ),  # tf.string means bytestring
        "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means scalar
        "one_hot_class": tf.io.VarLenFeature(tf.float32),
    }
    example = tf.io.parse_single_example(example, features)
    image = tf.image.decode_jpeg(example["image"], channels=3)
    image = (
        tf.cast(image, tf.float32) / 255.0
    )  # convert image to floats in [0, 1] range
    image = tf.reshape(
        image, [*IMAGE_SIZE, 3]
    )
    one_hot_class = tf.sparse.to_dense(example["one_hot_class"])
    one_hot_class = tf.reshape(one_hot_class, [5])
    return image, one_hot_class


def load_dataset(gcs_pattern, batch_size=32, training=True):
    filenames = tf.io.gfile.glob(gcs_pattern)
    ds = tf.data.TFRecordDataset(filenames).map(
        read_tfrecord).batch(batch_size)
    if training:
        return ds.repeat()
    else:
        return ds


def build_model():
    # MobileNet model for feature extraction
    mobilenet_v2 = 'https://tfhub.dev/google/imagenet/'\
        'mobilenet_v2_100_192/feature_vector/5'
    feature_extractor_layer = hub.KerasLayer(
        mobilenet_v2,
        input_shape=[*IMAGE_SIZE, 3],
        trainable=False
    )

    # Instantiate model
    model = tf.keras.Sequential([
        feature_extractor_layer,
        tf.keras.layers.Dense(5, activation="softmax")
    ])

    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    return model


def train_and_evaluate(train_data_path,
                       eval_data_path,
                       output_dir,
                       batch_size,
                       num_epochs,
                       train_examples):

    model = build_model()
    train_ds = load_dataset(gcs_pattern=train_data_path,
                            batch_size=batch_size)
    eval_ds = load_dataset(gcs_pattern=eval_data_path,
                           training=False)
    num_batches = batch_size * num_epochs
    steps_per_epoch = train_examples // num_batches

    history = model.fit(
        train_ds,
        validation_data=eval_ds,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        verbose=2,  # 0=silent, 1=progress bar, 2=one line per epoch
    )

    model.save(output_dir)

    print("Exported trained model to {}".format(output_dir))

if __name__ == "__main__":
    fire.Fire(train_and_evaluate)
