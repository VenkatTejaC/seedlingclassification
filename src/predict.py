import tensorflow as tf
import joblib

from src.preprocessing import preprocess_image


# rebuild architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(128,128,3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(12,activation='softmax')
])

# load trained weights
model.load_weights("models/plant_weights.weights.h5")

# load label names
classes = joblib.load("models/classes.pkl")


def predict(image):

    processed_image = preprocess_image(image)

    preds = model.predict(processed_image)

    class_index = preds.argmax()

    prediction = classes[class_index]

    return prediction