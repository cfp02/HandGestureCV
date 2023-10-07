from fake_mouse import FakeMouse

from header import DATA_DIR, WORKING_DIR, dataset_size, SPACEBAR, ESC, labels_dict
from inference_classifier import run_inference_classifier_novideo, load_model


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


def pinch_mouse_move(target_sign, last_sign, last_mouse_position, last_pinch_position, landmarks, fake_mouse: FakeMouse, multiplier):
    '''
    Moves mouse based on thumb position
    '''
    # Use thumb, index finger, middle finger landmarks, at indices 4, 8, 12
    # moves mouse around based on thumb position, uses last mouse and thumb position to calculate delta
    

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

    return last_mouse_position, last_pinch_position

def left_click(target_sign, current_sign, last_sign, prev_mouse_coords, fake_mouse: FakeMouse):
    if current_sign == target_sign and last_sign != target_sign:
        fake_mouse.left_click(prev_mouse_coords[0], prev_mouse_coords[1])
    else:
        pass

def right_click(target_sign, current_sign, last_sign, prev_mouse_coords, fake_mouse: FakeMouse):
    if current_sign == target_sign and last_sign != target_sign:
        fake_mouse.right_click(prev_mouse_coords[0], prev_mouse_coords[1])
    else:
        pass


def run_couch_potato_active(model, fake_mouse: FakeMouse):
    open_buffer:int = 0
    last_sign:str = '' 
    last_mouse_position = fake_mouse.get_mouse_coords()
    last_pinch_position = (0,0)

    active_bool = False
    waiting_for_stopstart_completion = False # after stopping, wait for change in sign to actually exit

    while True:
        current_sign, landmarks = run_inference_classifier_novideo(model, labels_dict)

        # Only look for reactivation if it is inactive and not still waiting for the stop (change from open to something else)
        if active_bool == False and waiting_for_stopstart_completion == False:

            match current_sign:
                case 'Open':
                    open_buffer += 1
                    if open_buffer > 10:
                        print("Open detected, starting")
                        active_bool = True
                        open_buffer = 0
                        waiting_for_stopstart_completion = True
                        continue
        
        else: # active_bool == True, so we are in active mode
            # Skip until hand sign changes from "Open" to something else after coming from a breaking condition using "Open"
            if waiting_for_stopstart_completion:
                if current_sign != 'Open':
                    waiting_for_stopstart_completion = False
                continue

            match current_sign:
                case 'Open':
                    open_buffer += 1
                    if open_buffer > 10:
                        print("Open detected, pausing")
                        active_bool = False
                        open_buffer = 0
                        waiting_for_stopstart_completion = True
                case 'Rah':
                    right_click(current_sign, current_sign, last_sign, last_mouse_position, fake_mouse)
                case 'Pinch':
                    last_mouse_position, last_pinch_position = pinch_mouse_move(current_sign, last_sign, last_mouse_position, last_pinch_position, landmarks, fake_mouse, 1000)
                case 'L':
                    left_click(current_sign, current_sign, last_sign, last_mouse_position, fake_mouse)
                case 'D':
                    pass
            
            if current_sign != 'Open':
                open_buffer = 0
                
            last_sign = current_sign

def main():
    fake_mouse = FakeMouse()
    model = load_model(WORKING_DIR + '/model_tflite.p')
    run_couch_potato_active(model, fake_mouse)
    
    #fix starting coords, yeets to the corner sometimes, might have to set init to current mouse position

    # Make instance of fake_mouse
    # Run classifier to get current hand symbol
    # Move mouse or act accordingly
    # Repeat

    fake_mouse.move_to(1920/2, 1080/2)

if __name__ == '__main__':
    main()