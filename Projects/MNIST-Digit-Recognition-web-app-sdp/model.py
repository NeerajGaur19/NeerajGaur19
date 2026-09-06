import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout


MODEL_PATH = "mnist_cnn_model.keras"


def create_and_train_model():
    """
    Create and train a CNN model on the MNIST dataset.
    """

    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize pixel values from 0-255 to 0-1
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Add channel dimension
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

    # Build CNN
    model = Sequential([
        Conv2D(
            32,
            kernel_size=(3, 3),
            activation="relu",
            input_shape=(28, 28, 1)
        ),

        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(
            64,
            kernel_size=(3, 3),
            activation="relu"
        ),

        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),

        Dense(128, activation="relu"),

        Dropout(0.3),

        Dense(10, activation="softmax")
    ])

    # Compile model
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Train model
    model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=128,
        validation_data=(x_test, y_test)
    )

    # Evaluate model
    loss, accuracy = model.evaluate(
        x_test,
        y_test,
        verbose=0
    )

    print(f"Test Accuracy: {accuracy:.4f}")

    # Save model
    model.save(MODEL_PATH)

    return model


if __name__ == "__main__":
    create_and_train_model()