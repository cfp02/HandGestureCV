import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import os
from datetime import datetime

from header import DATA_DIR, number_of_classes, dataset_size, SPACEBAR, ESC



def check_directories():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def record_photos(cap : cv2.VideoCapture):
    # Record video to get images to train each class
    cap = cv2.VideoCapture(0)
    for j in range(number_of_classes):
        if not os.path.exists(os.path.join(DATA_DIR, str(j))):
            os.makedirs(os.path.join(DATA_DIR, str(j)))

        print('Collecting data for class {}'.format(j))
        done = False
        while True:
            ret, frame = cap.read()
            cv2.putText(frame, 'Ready? Press "Space"', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                        cv2.LINE_AA)
            cv2.imshow('frame', frame)
            # Get key pressed
            key_pressed = cv2.waitKey(1) & 0xFF
            if key_pressed == SPACEBAR:
                print("Starting")
                break
            # If esc pressed
            elif key_pressed == ESC:
                print("Quitting")
                end_program()

        counter = 0
        while counter < dataset_size:
            ret, frame = cap.read()
            cv2.imshow('frame', frame)
            k = cv2.waitKey(25)
            if k == 27:
                print("Quitting")
                end_program()
                break
            cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)
            counter += 1

def end_program():
    cv2.destroyAllWindows()
    # exit()

def setup():
    check_directories()

def collect_images():
    cap : cv2.VideoCapture = None
    setup()
    record_photos(cap)

if __name__ == '__main__':
    collect_images()