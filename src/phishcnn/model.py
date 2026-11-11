"""Independent implementations of single- and multi-width character CNNs."""

import random
import numpy as np
from .urls import url_representation


def set_seed(seed):
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def text_dataset(texts, labels=None, batch_size=128, shuffle=False, seed=42):
    import tensorflow as tf
    # Shape (batch, 1) matches Keras Input(shape=(1,), dtype=tf.string).
    x = np.asarray([str(s) for s in texts], dtype=object).reshape(-1, 1)
    x = tf.constant(x.tolist(), dtype=tf.string)
    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(x)
    else:
        y = tf.constant(np.asarray(labels, dtype=np.float32).reshape(-1, 1))
        ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(min(len(x), 100000), seed=seed)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_cnn(train_texts, max_len, widths, config, seed):
    import tensorflow as tf
    set_seed(seed)
    vectorizer = tf.keras.layers.TextVectorization(
        standardize=None, split="character", output_mode="int",
        output_sequence_length=max_len,
    )
    vec_ds = text_dataset(train_texts, batch_size=2048)
    vectorizer.adapt(vec_ds)
    inp = tf.keras.Input(shape=(1,), dtype=tf.string, name="url_text")
    x = vectorizer(inp)
    x = tf.keras.layers.Embedding(
        input_dim=len(vectorizer.get_vocabulary()),
        output_dim=config["embedding_dim"], name="character_embedding",
    )(x)
    branches = []
    for width in widths:
        branch = tf.keras.layers.Conv1D(
            config["filters"], width, activation="relu",
            name=f"conv_width_{width}",
        )(x)
        branch = tf.keras.layers.GlobalMaxPooling1D(name=f"pool_width_{width}")(branch)
        branches.append(branch)
    x = branches[0] if len(branches) == 1 else tf.keras.layers.Concatenate()(branches)
    x = tf.keras.layers.Dense(config["dense_units"], activation="relu")(x)
    x = tf.keras.layers.Dropout(config["dropout"])(x)
    y = tf.keras.layers.Dense(1, activation="sigmoid", name="prob_phishing")(x)
    model = tf.keras.Model(inp, y, name="group7_character_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config["learning_rate"]),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="roc_auc"),
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
        ],
    )
    return model


def fit_cnn(model, train, val, representation, config, seed):
    import tensorflow as tf
    tr_text = train["URL"].map(lambda x: url_representation(x, representation))
    va_text = val["URL"].map(lambda x: url_representation(x, representation))
    return model.fit(
        text_dataset(tr_text, train["label"], config["batch_size"], shuffle=True, seed=seed),
        validation_data=text_dataset(va_text, val["label"], config["batch_size"]),
        epochs=config["epochs"], verbose=2,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_pr_auc", mode="max",
            patience=config["early_stopping_patience"], restore_best_weights=True,
        )],
    )


def probabilities(model, urls, representation, batch_size=128):
    texts = [url_representation(u, representation) for u in urls]
    return model.predict(text_dataset(texts, batch_size=batch_size), verbose=0).ravel()
