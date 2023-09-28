import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import os
from datetime import datetime

from header import DATA_DIR, dataset_size, SPACEBAR, ESC, labels_dict



def check_directories():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def record_photos_all_classes(cap : cv2.VideoCapture, labels_dict, dataset_size):
    # Record video to get images to train each class
    cap = cv2.VideoCapture(0)
    for j in range(len(labels_dict)):
        record_photos_this_class(cap, labels_dict[j], dataset_size)
        
def record_photos_this_class(cap : cv2.VideoCapture, which_class: str, dataset_size, append: bool = False):
    cap = cv2.VideoCapture(0)
    # Create the directory for this class if it doesn't exist
    this_label_path = os.path.join(DATA_DIR, which_class)
    if not os.path.exists(this_label_path):
        os.makedirs(this_label_path)

    # Initialize the data file counter, either 0 or the number of files already in the directory if appending
    counter = 0
    current_data_count = 0
    if append:
        print("Appending to existing data")
        counter = len(os.listdir(this_label_path))
        current_data_count = counter

    else:
        # Clear directory
        print("Overwriting existing data")
        for file in os.listdir(this_label_path):
            os.remove(os.path.join(this_label_path, file))

    print('Collecting data for class {}'.format(which_class))
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

    while counter < current_data_count + dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        k = cv2.waitKey(25)
        if k == ESC:
            print("Quitting")
            end_program()
            break
        cv2.imwrite(os.path.join(DATA_DIR, which_class, '{}.jpg'.format(counter)), frame)
        counter += 1

def end_program():
    cv2.destroyAllWindows()
    # exit()

def setup():
    check_directories()

def collect_images():
    cap : cv2.VideoCapture = None
    setup()
    # Options: Recapture data for all datasets, or specific dataset. Press 0 for all, 1-len(labels_dict) for specific
    all_str:str = 'A'
    print("\nPress {} to recapture data for all classes, or 0-{} to recapture data for a specific class".format(all_str, len(labels_dict)-1))
    # Print all classes with corresponding number on one line with list comprehension
    print(*["{}: {}".format(key, labels_dict[key]) for key in labels_dict], sep='\n')
    print("\n")
    class_index = input()
    this_label = ''
    if class_index.lower() != all_str.lower():
        this_label = labels_dict[int(class_index)]
    print("You chose {}".format(this_label))
    print("Overwrite or append? Press 0 to overwrite, 1 to append")
    append = False
    append_ = input()
    if append_ == '1':
        append = True

    if class_index == all_str or class_index == all_str.lower():
        record_photos_all_classes(cap, labels_dict, dataset_size)
    elif this_label in labels_dict.values():
        record_photos_this_class(cap, this_label, dataset_size, append)
    else:
        print("Invalid input, exiting")
        end_program()

    cv2.destroyAllWindows()


    # record_photos_all_classes(cap, labels_dict, dataset_size)

if __name__ == '__main__':
    collect_images()