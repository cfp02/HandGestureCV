from fake_mouse import FakeMouse

from header import DATA_DIR, WORKING_DIR, number_of_classes, dataset_size, SPACEBAR, ESC, labels_dict
from inference_classifier import run_inference_classifier_novideo, load_model




def main():
    fake_mouse = FakeMouse()
    model = load_model(WORKING_DIR + '/model.p')
    open_buffer = 0
    while True:
        current_sign = run_inference_classifier_novideo(model, labels_dict)
        print(current_sign)
        if current_sign == 'Open':
            open_buffer += 1
        else:
            open_buffer = 0
        if open_buffer > 5:
            print("Open detected")
            break

    # Make instance of fake_mouse
    # Run classifier to get current hand symbol
    # Move mouse or act accordingly
    # Repeat

    fake_mouse.move_to(100,100)

if __name__ == '__main__':
    main()