
import cv2
import mediapipe as mp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
from mediapipe.python.solutions import hands as mp_hands, drawing_utils as mp_drawing, drawing_styles as mp_drawing_styles

from header import DATA_DIR, WORKING_DIR, number_of_classes, dataset_size, SPACEBAR, ESC

def load_model(model_path):
    model_dict = pickle.load(open(model_path, 'rb'))
    model : RandomForestClassifier = model_dict['model']
    return model

def run_classifier(model:RandomForestClassifier, labels_dict, hands: mp_hands.Hands, cap: cv2.VideoCapture):
    while True:
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            if(len(results.multi_hand_landmarks) > 1):
                print("More than one hand detected")
                continue
            for hand_landmarks in results.multi_hand_landmarks:
                draw_landmarks(frame, hand_landmarks)
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

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            prediction = model.predict(np.array([data_aux]))
            certainty = model.predict_proba(np.array([data_aux]))
            # Print percent certainty by turning certainty into a single percentage from the array
            # print(certainty[0][int(prediction[0])] * 100)
            # print(prediction)

            predicted_character = labels_dict[int(prediction[0])]

            draw_bounding_box(frame, predicted_character, x1, y1, x2, y2)
        
        cv2.imshow('frame', frame)
        k = cv2.waitKey(1)
        if k == ESC:
            end_program()
            break

def end_program():
    cv2.destroyAllWindows()
    # exit()

def draw_landmarks(frame, hand_landmarks):
    mp_drawing.draw_landmarks(
        frame,  # image to draw
        hand_landmarks,  # model output
        mp_hands.HAND_CONNECTIONS,  # hand connections
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style())
    
def draw_bounding_box(frame, predicted_character, x1, y1, x2, y2):
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
    cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)


def inference_classifier():
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
    model = load_model(WORKING_DIR + '/model.p')
    labels_dict = {0: 'QC', 1: 'L', 2: 'Pinch', 3: 'Open', 4: 'Point'}
    cap = cv2.VideoCapture(0)
    run_classifier(model, labels_dict, hands, cap)

if __name__ == '__main__':
    inference_classifier()