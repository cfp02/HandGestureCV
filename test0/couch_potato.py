from fake_mouse import FakeMouse

from header import DATA_DIR, WORKING_DIR, number_of_classes, dataset_size, SPACEBAR, ESC, labels_dict
from inference_classifier import run_inference_classifier_novideo, load_model

open_buffer :int = 0
last_sign = ''
last_mouse_position = (0,0) # (x,y)
last_pinch_position = (0,0) # (x,y)

def open_detection(current_sign, buffer_len = 5):
    global open_buffer
    if current_sign == 'Open':
        open_buffer += 1
    else:
        open_buffer = 0
    if open_buffer > buffer_len:
        print("Open detected")
        return True
    return False

def pinch_mouse_move(current_sign, last_sign, landmarks, fake_mouse: FakeMouse, multiplier):
    global last_mouse_position, last_pinch_position
    # Use THUMB_TIP landmark, at index 4
    # moves mouse around based on thumb position, uses last mouse and thumb position to calculate delta
    if current_sign == 'Pinch':
        thumb_tip = landmarks[4]
        thumb_tip_x = thumb_tip.x
        thumb_tip_y = thumb_tip.y
        if last_mouse_position == (0,0) or (last_sign != 'Pinch'):
            current_mouse = fake_mouse.get_mouse_coords()
            last_mouse_position = (current_mouse.x, current_mouse.y)
        if last_pinch_position == (0,0) or (last_sign != 'Pinch'):
            last_pinch_position = (thumb_tip_x, thumb_tip_y)
        delta_x = (thumb_tip_x - last_pinch_position[0]) * multiplier
        delta_y = (thumb_tip_y - last_pinch_position[1]) * multiplier
        fake_mouse.move_delta(delta_x, delta_y)
        current_mouse = fake_mouse.get_mouse_coords()
        last_mouse_position = (current_mouse.x, current_mouse.y)
        last_pinch_position = (thumb_tip_x, thumb_tip_y)
        

def main():
    global last_sign
    fake_mouse = FakeMouse()
    model = load_model(WORKING_DIR + '/model.p')
    while True:
        current_sign, landmarks = run_inference_classifier_novideo(model, labels_dict)
        # print(current_sign)
        if open_detection(current_sign, 10):
            print("Open detected")
            break

        if current_sign == 'Pinch':
            pinch_mouse_move(current_sign, last_sign, landmarks, fake_mouse, 1000)
        
        last_sign = current_sign


    # Make instance of fake_mouse
    # Run classifier to get current hand symbol
    # Move mouse or act accordingly
    # Repeat

    fake_mouse.move_to(1920/2, 1080/2)

if __name__ == '__main__':
    main()