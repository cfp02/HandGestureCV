import pyautogui

class FakeMouse:
    def __init__(self):
        pass

    def get_mouse_coords(self):
        return pyautogui.position()

    def move_to(self, x, y):
        pyautogui.moveTo(x, y)
        print("Mouse move: ", x, y)

    def move_delta(self, x, y):
        pyautogui.move(x, y)
        print("Mouse move delta: ", x, y)

    def click(self, x, y):
        print("Mouse click: ", x, y)

    def press(self, button):
        print("Mouse press: ", button)

    def release(self, button):
        print("Mouse release: ", button)

    def scroll(self, x, y):
        print("Mouse scroll: ", x, y)