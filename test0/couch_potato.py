from fake_mouse import FakeMouse

from header import DATA_DIR, WORKING_DIR, dataset_size, SPACEBAR, ESC, labels_dict
from inference_classifier import run_inference_classifier_novideo, load_model

open_buffer :int = 0
last_sign = ''
last_mouse_position = (0,0) # (x,y)
last_pinch_position = (0,0) # (x,y)

def open_detection(current_sign, buffer_len = 5):
    '''
    Detects if hand is open for a certain number of frames
    '''
    
    global open_buffer
    if current_sign == 'Open':
        open_buffer += 1
    else:
        open_buffer = 0
    if open_buffer > buffer_len:
        print("Open detected")
        return True
    return False


def pinch_mouse_move(target_sign, current_sign, last_sign, landmarks, fake_mouse: FakeMouse, multiplier):
    '''
    Moves mouse based on thumb position
    '''

    global last_mouse_position, last_pinch_position
    # Use thumb, index finger, middle finger landmarks, at indices 4, 8, 12
    # moves mouse around based on thumb position, uses last mouse and thumb position to calculate delta
    
    if current_sign == target_sign:
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        thumb_tip_x, thumb_tip_y = thumb_tip.x, thumb_tip.y
        index_tip_x, index_tip_y = index_tip.x, index_tip.y
        middle_tip_x, middle_tip_y = middle_tip.x, middle_tip.y
        average_x = (thumb_tip_x + index_tip_x + middle_tip_x) / 3
        average_y = (thumb_tip_y + index_tip_y + middle_tip_y) / 3
        if last_mouse_position == (0,0) or (last_sign != target_sign):
            current_mouse = fake_mouse.get_mouse_coords()
            last_mouse_position = (current_mouse.x, current_mouse.y)
        if last_pinch_position == (0,0) or (last_sign != target_sign):
            last_pinch_position = (average_x, average_y)
        delta_x, delta_y = (average_x - last_pinch_position[0]) * multiplier, (average_y - last_pinch_position[1]) * multiplier
        # Move the mouse by delta_x, delta_y
        fake_mouse.move_delta(delta_x, delta_y)
        current_mouse = fake_mouse.get_mouse_coords()
        last_mouse_position = (current_mouse.x, current_mouse.y)
        last_pinch_position = (average_x, average_y)

def left_click(target_sign, current_sign, last_sign, prev_mouse_coords, fake_mouse: FakeMouse):
    if current_sign == target_sign and last_sign != target_sign:
        fake_mouse.left_click(prev_mouse_coords[0], prev_mouse_coords[1])
    else:
        pass

def main():
    global last_sign, last_mouse_position
    fake_mouse = FakeMouse()
    model = load_model(WORKING_DIR + '/model.p')
    while True:
        current_sign, landmarks = run_inference_classifier_novideo(model, labels_dict)
        # print(current_sign)
        if open_detection(current_sign, 10):
            print("Open detected")
            break

        match current_sign:
            case 'Open':
                pass
            case 'Rah':
                pass
            case 'Pinch':
                pinch_mouse_move(current_sign, current_sign, last_sign, landmarks, fake_mouse, 1000)
            case 'L':
                left_click(current_sign, current_sign, last_sign, last_mouse_position, fake_mouse)
            case 'D':
                pass
            
           
        
        last_sign = current_sign


    # Make instance of fake_mouse
    # Run classifier to get current hand symbol
    # Move mouse or act accordingly
    # Repeat

    fake_mouse.move_to(1920/2, 1080/2)

if __name__ == '__main__':
    main()