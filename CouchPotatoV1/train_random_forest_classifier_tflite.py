import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow_decision_forests as tfdf
import pandas as pd

from header import WORKING_DIR

# Load your data
def load_data(data_path=WORKING_DIR + '\\landmark_dataset_tflite.pickle'):
    data_dict = pickle.load(open(data_path, 'rb'))
    landmark_data = np.array(data_dict['data'])
    landmark_labels = np.array(data_dict['labels'])
    return landmark_data, landmark_labels

# Create test and train split
def create_test_train_split(data, labels, test_size=0.2):
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, shuffle=True, stratify=labels)
    return x_train, x_test, y_train, y_test

def train_random_forest_classifier():
    landmark_data, landmark_labels = load_data()
    x_train, x_test, y_train, y_test = create_test_train_split(landmark_data, landmark_labels)

    # Normalize input data (assuming your data is in the range [0, 255])
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Convert data to a DataFrame
    train_df = pd.DataFrame(data=x_train, columns=[f"feature_{i}" for i in range(x_train.shape[1])])
    train_df["label"] = y_train

    # Define a Random Forest classifier using tensorflow_decision_forests
    task = tfdf.keras.premade.Task(
        label="label",
        # Change the task type based on your task (e.g., CLASSIFICATION or REGRESSION)
        # For classification, you can use tfdf.keras.premade.ClassificationTask
        # For regression, you can use tfdf.keras.premade.RegressionTask
        task_type=tfdf.keras.premade.TaskType.CLASSIFICATION,
    )

    # Create and train the Random Forest model
    model = tfdf.keras.RandomForestModel(task=task)
    model.compile(metrics=["accuracy"])
    train_data = train_df.copy()
    train_data["label"] = train_data["label"].astype("string")  # Ensure labels are in string format for classification
    model.fit(x=train_data)

    # Evaluate the model
    test_df = pd.DataFrame(data=x_test, columns=[f"feature_{i}" for i in range(x_test.shape[1])])
    predictions = model.predict(test_df)
    y_pred = predictions["class_ids"]
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

if __name__ == '__main__':
    train_random_forest_classifier()
