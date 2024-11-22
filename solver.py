import random
import pyautogui
from pywinauto import Desktop
import cv2
import numpy as np

top_left = (0,0)
top_right = (0,0)
bottom_left = (0,0)
bottom_right = (0,0)

def take_screenshot():
    # Find the application window by title
    windows = Desktop(backend="uia").windows(title_re=".*Minesweeper Online")  # Adjust the title pattern
    if windows:
        app_window = windows[0]

        # Get window coordinates
        rect = app_window.rectangle()
        region = (rect.left, rect.top, rect.width(), rect.height())
        # Take a screenshot of the application window
        screenshot = pyautogui.screenshot(region=region)

        # Convert to NumPy array and OpenCV format
        screenshot_np = np.array(screenshot)
        screenshot_bgr = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)

        # Display and save the screenshot
        # cv2.imshow("Application Screenshot", screenshot_bgr)
        cv2.imwrite("screenshot.png", screenshot_bgr)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
def mouse_click(x, y):
    print(f"Detected region center: ({x}, {y})")
    # Simulate mouse click at the center of the detected region
    pyautogui.moveTo(x, y, duration=0)  # Move mouse with a slight delay
    pyautogui.click()  # Perform left-click
    
def mouse_move(x, y):
    # Simulate mouse click at the center of the detected region
    pyautogui.moveTo(x, y, duration=0)  # Move mouse with a slight delay
    
def mouse_click_random(top_left, bottom_right):
    top_left = (top_left[0] - 9, top_left[1] - 100 - 6)
    print(top_left[0], top_left[1])
    for i in range(5):
        x_init = top_left[0] + 17.5
        x_end = top_left[0] + 560 - 17.5
        y_start = top_left[1] + 17.5
        y_end = top_left[1] + 560 - 17.5
        
        x = random_between(x_init, x_end)
        y = random_between(y_start, y_end)
        print(x,y)
        # Simulate mouse click at the center of the detected region
        pyautogui.moveTo(x,y, duration=0)  # Move mouse with a slight delay
        pyautogui.click()  # Perform left-click
        take_screenshot()
        reset_if_dead()
   
def random_between(a, b):
    # Ensure a is the smaller number and b is the larger number
    if a > b:
        a, b = b, a  # Swap if necessary
    a = int(a)
    b = int(b)
    # Generate a random integer between a and b (inclusive)
    return random.randint(a, b) 
        
def template_matching():
    screenshot = cv2.imread("screenshot.png", cv2.IMREAD_GRAYSCALE)
    template = cv2.imread("template_intermediate.png", cv2.IMREAD_GRAYSCALE)

    if screenshot is None or template is None:
        print("Error: Could not load the screenshot or template. Please check the file paths.")
    else:
        # Perform template matching
        result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)

        # Get the best match position
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Extract the top-left corner of the matching region
        top_left = max_loc
        h, w = template.shape  # Template dimensions
        bottom_right = (top_left[0] + w, top_left[1] + h)

        # Draw a rectangle around the detected area
        screenshot_with_box = cv2.cvtColor(screenshot, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(screenshot_with_box, top_left, bottom_right, (0, 255, 0), 2)
        top_left = (top_left[0] + 18, top_left[1] + 114)
        top_right = (top_left[0] + 18 + 560, top_left[1] + 114)
        bottom_left = (top_left[0] + 18, top_left[1] + 114 + 560)
        bottom_right = (top_left[0] + 18 + 560, top_left[1] + 114 + 560)
        
        mouse_click_random((top_left[0] + 18, top_left[1] + 114), (top_left[0] + 18 + 560, top_left[1] + 114 + 560))
        # Crop the detected region
        cropped_region = screenshot[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        # Calculate width and height
        width = bottom_right[0] - top_left[0]
        height = bottom_right[1] - top_left[1]
        print(f'{height} x {width}')

        # Show the results
        # cv2.imshow("Detected Template", screenshot_with_box)
        # cv2.imshow("Cropped Region", cropped_region)
        cv2.imwrite("detected_template.png", screenshot_with_box)
        cv2.imwrite("cropped_region.png", cropped_region)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

def reset_if_dead():
    screenshot = cv2.imread("screenshot.png", cv2.IMREAD_GRAYSCALE)
    template = cv2.imread("unhappy.png", cv2.IMREAD_GRAYSCALE)

    if screenshot is None or template is None:
        print("Error: Could not load the screenshot or template. Please check the file paths.")
    else:
        # Perform template matching
        result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)

        # Get the best match position
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Extract the top-left corner of the matching region
        top_left = max_loc
        h, w = template.shape  # Template dimensions
        bottom_right = (top_left[0] + w, top_left[1] + h)

        # If the max value is above the threshold, it's considered a match
        if max_val >= 0.9:
            # Draw a rectangle around the detected area
            screenshot_with_box = cv2.cvtColor(screenshot, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(screenshot_with_box, top_left, bottom_right, (0, 255, 0), 2)
            # # top left
            x = top_left[0] + 17.5
            y = top_left[1] + 17.5
            mouse_click(x, y)
            return True
        return False

take_screenshot()
template_matching()
# reset_if_dead()