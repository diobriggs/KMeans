import ray
import numpy as np
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
import tensorflow as tf

ray.init()

def load_data():
    digits = datasets.load_digits()
    # Reshape images to (8, 8)
    images = digits.images.reshape((len(digits.images), 8, 8))
    return images, digits.target

@ray.remote
def train_and_predict(X_train, y_train, X_test):
    # Define and compile the neural network model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(8, 8)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    # Train the model
    model.fit(X_train, y_train, epochs=5)
    
    # Predict on the test set
    predicted_probs = model.predict(X_test)
    predicted = tf.argmax(predicted_probs, axis=1).numpy()
    
    return predicted

if __name__ == "__main__":
    # Load data
    X_data, y_data = load_data()

    # Split data into train and test subsets
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.3, shuffle=False
    )

    # Distribute data to the cluster
    X_train_id = ray.put(X_train)
    y_train_id = ray.put(y_train)
    X_test_id = ray.put(X_test)

    # Train and predict in parallel
    predicted_ids = [train_and_predict.remote(X_train_id, y_train_id, X_test_id)]

    # Get predicted values
    predicted = ray.get(predicted_ids)[0]

    # Compute metrics
    classification_report = metrics.classification_report(y_test, predicted)
    confusion_matrix = metrics.confusion_matrix(y_test, predicted)

    print(f"Classification report for classifier:\n{classification_report}\n")
    print(f"Confusion matrix:\n{confusion_matrix}")

    ray.shutdown()
