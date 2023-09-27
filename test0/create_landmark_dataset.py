import os

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
from mediapipe.python.solutions import hands as mp_hands, drawing_utils as mp_drawing, drawing_styles as mp_drawing_styles

from header import DATA_DIR, WORKING_DIR, number_of_classes, dataset_size, SPACEBAR, ESC

# mp.solutions imports other packages, so links to the hands class do not work with VS code
# mp_hands = hands
# mp_drawing = drawing_utils
# mp_drawing_styles = drawing_styles

def create_landmark_dataset():

    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    landmark_dataset = []
    landmark_labels = []

    # For each class directory
    for dir_ in os.listdir(DATA_DIR):
        # For each image in the class folder
        for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):

            data_aux = []
            x_ = []
            y_ = []

            img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            from mediapipe.framework.formats import landmark_pb2

            results= hands.process(img_rgb)
            if results.multi_hand_landmarks:
                if(len(results.multi_hand_landmarks) > 1):
                    print("More than one hand detected in " + img_path)
                    continue
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                landmark_dataset.append(data_aux)
                landmark_labels.append(dir_)

    print("Finished creating landmark dataset")

    # for i in range(len(landmark_dataset)):
    #     print(i, len(landmark_dataset[i]))

    import pickle
    with open(WORKING_DIR + '\\landmark_dataset.pickle', 'wb') as f:
        pickle.dump({'data': landmark_dataset, 'labels': landmark_labels}, f)

    print("Wrote landmark dataset to " + WORKING_DIR + '\\landmark_dataset.pickle')