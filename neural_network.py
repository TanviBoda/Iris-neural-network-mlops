import joblib
import mlflow
mlflow.set_experiment("Iris_Neural_Network")
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


# 1. Load Iris dataset from CSV
df = pd.read_csv("iris.csv")

X = df.drop("species", axis=1).values

encoder = LabelEncoder()
y = encoder.fit_transform(df["species"])

print("Dataset Shape:")
print(X.shape)

print("\nClasses:")
print(encoder.classes_)


# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=10,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# 3. Standardize data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
joblib.dump(scaler, "scaler.pkl")
X_test = scaler.transform(X_test)


# 4. Create Neural Network
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(3, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# MLflow experiment tracking
with mlflow.start_run():
    # Log model parameters
    mlflow.log_param("epochs", 40)
    mlflow.log_param("batch_size", 8)
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("hidden_layer_1", 32)
    mlflow.log_param("hidden_layer_2", 16)
    mlflow.log_param("optimizer", "Adam")

    # Train the model
    history = model.fit(
        X_train,
        y_train,
        epochs=40,
        batch_size=8,
        validation_split=0.2,
        verbose=1
    )

    # Evaluate on the test dataset
    loss, accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )
    
    # Validate model accuracy
    MINIMUM_ACCURACY = 0.90

    print(f"\nTest accuracy: {accuracy:.2%}")
    print(f"Minimum required accuracy: {MINIMUM_ACCURACY:.2%}")

    if accuracy < MINIMUM_ACCURACY:
        raise ValueError(
            f"Model accuracy {accuracy:.2%} is below "
            f"the required {MINIMUM_ACCURACY:.2%}"
        )

    print("Model accuracy validation passed!")


    # Log test metrics
   
mlflow.log_metric("test_accuracy", float(accuracy))
mlflow.log_metric("test_loss", float(loss))

# Log final training and validation metrics
mlflow.log_metric(
    "final_training_accuracy",
    float(history.history["accuracy"][-1])
)
mlflow.log_metric(
    "final_validation_accuracy",
    float(history.history["val_accuracy"][-1])
)
mlflow.log_metric(
    "final_training_loss",
    float(history.history["loss"][-1])
)
mlflow.log_metric(
    "final_validation_loss",
    float(history.history["val_loss"][-1])
)
model.save("iris_model.keras")
# Log the trained TensorFlow model
mlflow.tensorflow.log_model(model, name="iris_neural_network")



print("\nTest Accuracy:")
print(accuracy)

print("\nTest Accuracy Percentage:")
print(accuracy * 100, "%")


# 9. Predictions
probabilities = model.predict(X_test)

y_pred = np.argmax(
    probabilities,
    axis=1
)


# 10. Confusion Matrix
cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# 11. Display Confusion Matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=encoder.classes_
)

disp.plot()
plt.title("Iris Classification using TensorFlow Neural Network")
plt.savefig("confusion_matrix.png")
plt.close()


# 12. Accuracy graph
plt.figure()
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("accuracy.png")
plt.close()

# 13. Loss graph
plt.figure()
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss.png")
plt.close()