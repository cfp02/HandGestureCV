
from collect_images import collect_images
from create_landmark_dataset import create_landmark_dataset
from train_random_forest_classifier import train_random_forest_classifier
from inference_classifier import inference_classifier


def main():
    # Choices: collect images, train model, test model
    while True:
        print("\nCollect Images: 1\nTrain Model: 2\nTest Model: 3\n")
        answer = input()
        if answer == '1':
            collect_images()
        elif answer == '2':
            create_landmark_dataset()
            train_random_forest_classifier()
        elif answer == '3':
            inference_classifier()
        else:
            print("Other, exiting")
            exit()

if __name__ == '__main__':
    main()