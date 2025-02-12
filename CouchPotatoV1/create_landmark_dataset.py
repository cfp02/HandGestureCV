

# Load the TFLite model for hand landmark prediction and use it in place of mediapipe.

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from header import DATA_DIR, WORKING_DIR, dataset_size, SPACEBAR, ESC

path_to_model = WORKING_DIR + '\\hand_landmark_full.tflite'

# Load your TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path=path_to_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# print(input_details)
# print(output_details)

# Function to preprocess and get hand landmarks from an image
def get_hand_landmarks(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print(img_rgb.shape)
    # print(input_details[0]['shape'])

    # Preprocess the image if needed to match the model input requirements
    # Input dimensions for image need to be 224x224, so crop the image to a square
    img_rgb_squarecrop = img[0:480,80:560]
    img_rgb_squarecrop = cv2.resize(img_rgb_squarecrop, (224, 224))
    img_rgb_squarecrop = np.expand_dims(img_rgb_squarecrop, axis=0).astype(np.float32)
    # print(img_rgb_squarecrop.shape)
    # print(img_rgb_squarecrop.shape)
    cv2.waitKey(0)

    # Perform inference with the TensorFlow Lite model
    interpreter.set_tensor(0, img_rgb_squarecrop)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Extract hand landmarks from the model output
    # You'll need to adapt this part based on your model's output format
    hand_landmarks = process_output_data(output_data)

    return hand_landmarks

# Function to process the model's output and extract hand landmarks
def process_output_data(output_data):
    # Adapt this part to extract hand landmarks based on your model's output format
    # This will depend on how your model was trained and what it outputs
    # Typically, you'll need to parse the output tensor to get the landmarks
    # For example, you might need to reshape and scale the output data

    hand_landmarks = output_data  # Modify this line to match your model's output format

    return hand_landmarks

def create_landmark_dataset():
    landmark_dataset = []
    landmark_labels = []
    # For each class directory
    for dir_ in os.listdir(DATA_DIR):
        # For each image in the class folder
        for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
            image_path = os.path.join(DATA_DIR, dir_, img_path)

            # Get hand landmarks using your TensorFlow Lite model
            hand_landmarks = get_hand_landmarks(image_path)[0]

            if hand_landmarks is not None:
                landmark_dataset.append(hand_landmarks)
                landmark_labels.append(dir_)

    print("Finished creating landmark dataset")

    # Save the dataset or use it for further processing

    import pickle
    with open(WORKING_DIR + '\\landmark_dataset_tflite.pickle', 'wb') as f:
        pickle.dump({'data': landmark_dataset, 'labels': landmark_labels}, f)

    print("Wrote landmark dataset to " + WORKING_DIR + '\\landmark_dataset_tflite.pickle')

# Call the function to create the landmark dataset
create_landmark_dataset()
