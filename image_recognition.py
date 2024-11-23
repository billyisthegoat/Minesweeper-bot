import cv2
import numpy as np
import pyautogui

def find_template_top_expert():
    # Take a screenshot
    screenshot = pyautogui.screenshot()

    # Convert the screenshot to a NumPy array and BGR format
    image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    # Load the template
    template = cv2.imread("expert.png", cv2.IMREAD_COLOR)

    # Convert both images to grayscale for matching
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Match the template
    result = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED)

    # Find the location with the maximum match
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    # The top-left corner of the matched region
    top_left = max_loc
    print(f"Match found at: {top_left} with confidence: {max_val}")

    return top_left

def find_template_top_intermediate():
    # Take a screenshot
    screenshot = pyautogui.screenshot()

    # Convert the screenshot to a NumPy array and BGR format
    image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    # Load the template
    template = cv2.imread("intermediate.png", cv2.IMREAD_COLOR)

    # Convert both images to grayscale for matching
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Match the template
    result = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED)

    # Find the location with the maximum match
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    # The top-left corner of the matched region
    top_left = max_loc
    print(f"Match found at: {top_left} with confidence: {max_val}")

    return top_left