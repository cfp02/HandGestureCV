import pickle
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from micromlgen import port

from header import DATA_DIR, WORKING_DIR, dataset_size, SPACEBAR, ESC


def load_data(data_path = WORKING_DIR + '\\landmark_dataset_tflite.pickle'):
    data_dict = pickle.load(open(data_path, 'rb'))
    landmark_data = np.array(data_dict['data'])
    landmark_labels = np.array(data_dict['labels'])
    return landmark_data, landmark_labels

def create_test_train_split(data, labels, test_size=0.2):
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, shuffle=True, stratify=labels)
    return x_train, x_test, y_train, y_test

def train_classifier(x_train, y_train):
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    return model

def test_classifier(model: RandomForestClassifier, x_test, y_test):
    y_predict = model.predict(x_test)
    score = accuracy_score(y_predict, y_test)
    print('{}% of samples were classified correctly !'.format(score * 100))

def save_model(model, model_path = WORKING_DIR + '\\random_forest_model.p'):
    f = open(model_path, 'wb')
    pickle.dump({'model': model}, f)
    f.close()

# Creates .h model for use on an Arduino
def export_model_to_arduino(model, model_path = WORKING_DIR + '\\random_forest_model.h'):
    random_forest_arduino_c = port(model)
    with open(model_path, 'w') as f:
        f.write(random_forest_arduino_c)
        print("Model exported to " + model_path + " for use on an Arduino")

def train_random_forest_classifier():
    landmark_data, landmark_labels = load_data()
    x_train, x_test, y_train, y_test = create_test_train_split(landmark_data, landmark_labels)
    model = train_classifier(x_train, y_train)
    test_classifier(model, x_test, y_test)
    save_model(model)
    export_model_to_arduino(model)

if __name__ == '__main__':
    train_random_forest_classifier()