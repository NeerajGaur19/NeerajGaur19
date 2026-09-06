import tensorflow as tf

from pathlib import Path

from tensorflow.keras.datasets import mnist

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)


# ============================================================
# MODEL PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "mnist_cnn_model.keras"


# ============================================================
# CREATE AND TRAIN MODEL
# ============================================================

def create_and_train_model():

    print("\n")
    print("=" * 60)
    print("MNIST CNN TRAINING")
    print("=" * 60)

    # --------------------------------------------------------
    # LOAD MNIST
    # --------------------------------------------------------

    print("\nLoading MNIST dataset...")

    (
        x_train,
        y_train
    ), (
        x_test,
        y_test
    ) = mnist.load_data()

    print(
        f"Training samples: {len(x_train)}"
    )

    print(
        f"Testing samples: {len(x_test)}"
    )

    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    x_train = (
        x_train.astype("float32") / 255.0
    )

    x_test = (
        x_test.astype("float32") / 255.0
    )

    # --------------------------------------------------------
    # ADD CHANNEL DIMENSION
    # --------------------------------------------------------

    x_train = x_train.reshape(
        x_train.shape[0],
        28,
        28,
        1
    )

    x_test = x_test.reshape(
        x_test.shape[0],
        28,
        28,
        1
    )

    # --------------------------------------------------------
    # BUILD CNN
    # --------------------------------------------------------

    model = Sequential([

        # Convolution Layer 1
        Conv2D(
            32,
            kernel_size=(3, 3),
            activation="relu",
            input_shape=(28, 28, 1)
        ),

        # Pooling
        MaxPooling2D(
            pool_size=(2, 2)
        ),

        # Convolution Layer 2
        Conv2D(
            64,
            kernel_size=(3, 3),
            activation="relu"
        ),

        # Pooling
        MaxPooling2D(
            pool_size=(2, 2)
        ),

        # Flatten
        Flatten(),

        # Fully Connected Layer
        Dense(
            128,
            activation="relu"
        ),

        # Dropout
        Dropout(
            0.3
        ),

        # Output Layer
        Dense(
            10,
            activation="softmax"
        )
    ])

    # --------------------------------------------------------
    # COMPILE
    # --------------------------------------------------------

    model.compile(

        optimizer="adam",

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]
    )

    # --------------------------------------------------------
    # DISPLAY MODEL
    # --------------------------------------------------------

    print("\n")
    model.summary()

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print("\n")
    print("Training CNN...")

    history = model.fit(

        x_train,

        y_train,

        epochs=5,

        batch_size=128,

        validation_data=(
            x_test,
            y_test
        )
    )

    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------

    loss, accuracy = model.evaluate(

        x_test,

        y_test,

        verbose=0
    )

    print("\n")
    print("=" * 60)
    print(
        f"Test Loss     : {loss:.4f}"
    )
    print(
        f"Test Accuracy : {accuracy:.4f}"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    model.save(
        MODEL_PATH
    )

    print("\n")
    print(
        f"Model saved to:"
    )

    print(
        MODEL_PATH
    )

    return model, history


# ============================================================
# RUN TRAINING
# ============================================================

if __name__ == "__main__":

    create_and_train_model()