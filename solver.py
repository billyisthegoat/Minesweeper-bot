from concurrent.futures import ThreadPoolExecutor
import random
import time
import pyautogui
from pywinauto import Desktop
import cv2
import numpy as np
actions_taken = 1

top_left = (610,450)
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
    
def mouse_click_random():
    global top_left
    print(top_left[0], top_left[1])
    for i in range(5):
        x_init = top_left[0] + 17.5
        x_end = top_left[0] + 560 - 17.5
        y_start = top_left[1] + 17.5
        y_end = top_left[1] + 560 - 17.5
        
        x = random_between(x_init, x_end)
        y = random_between(y_start, y_end)
        print(x,y)
        pyautogui.moveTo(x,y, duration=0)
        pyautogui.click()
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
    global top_left
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
        # top_left = max_loc
        print(f"maxloc {max_loc}")
        h, w = template.shape  # Template dimensions
        bottom_right = (top_left[0] + w, top_left[1] + h)

        # Draw a rectangle around the detected area
        screenshot_with_box = cv2.cvtColor(screenshot, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(screenshot_with_box, top_left, bottom_right, (0, 255, 0), 2)
        # top_left = (top_left[0] + 18, top_left[1] + 114)
        print(f"Top left: {top_left}")
        bottom_right = (top_left[0] + 18 + 560, top_left[1] + 114 + 560)
        
        mouse_click_random()
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

# grid
def right_click(row,col):
    global actions_taken
    # 0,0 = +16, +16 (mid)
    # 1,1 = +16+35, +16+35
    # 2,0 = +16, +16 + 70
    x = top_left[0] + 16 + 35*row
    y = top_left[1] + 16 + 35*col
    pyautogui.moveTo(x, y, duration=0)  # Move mouse with a slight delay
    pyautogui.rightClick()  # Perform left-click
    actions_taken += 1


def left_click(row,col):
    global actions_taken
    # 0,0 = +16, +16 (mid)
    # 1,1 = +16+35, +16+35
    # 2,0 = +16, +16 + 70
    x = top_left[0] + 16 + 35*row
    y = top_left[1] + 16 + 35*col
    pyautogui.moveTo(x, y, duration=0)  # Move mouse with a slight delay
    pyautogui.leftClick()  # Perform left-click
    actions_taken += 1
    
# take_screenshot()
# template_matching()
# reset_if_dead()
grid = [[0] * 16 for _ in range(16)]

# def read_values():
#     coordinates = top_left
#     mid_point = 
    
def take_single_box(x,y):
    # Get window coordinates
    region = (x-10,y-10,35,35)
    # Take a screenshot of the application window
    screenshot = pyautogui.screenshot(region=region)

    # Convert to NumPy array and OpenCV format
    screenshot_np = np.array(screenshot)
    screenshot_bgr = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)

    # Display and save the screenshot
    # cv2.imshow("Application Screenshot", screenshot_bgr)
    cv2.imwrite("box.png", screenshot_bgr)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
def read_grid():
    start_time = time.time()  # Record start time

    # Capture the entire grid in one screenshot
    grid_width = 16 * 35
    grid_height = 16 * 35
    region = (top_left[0] - 8, top_left[1] - 8, grid_width, grid_height)
    full_screenshot = pyautogui.screenshot(region=region)

    # Convert to NumPy array and OpenCV format
    full_screenshot_np = np.array(full_screenshot)
    full_screenshot_bgr = cv2.cvtColor(full_screenshot_np, cv2.COLOR_RGB2BGR)

    # Precompute slicing coordinates
    slices = [(i, j, full_screenshot_bgr[35*i:35*(i+1), 35*j:35*(j+1)])
              for i in range(16) for j in range(16)]

    # Use parallel processing to get values
    def process_slice(data):
        i, j, img = data
        return i, j, get_value(img)

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_slice, slices))

    # Populate the grid with results
    for i, j, value in results:
        grid[i][j] = value

    # Print the grid
    for row in grid:
        print(" ".join(f"{str(val).rjust(2)}" for val in row))

    end_time = time.time()  # Record end time
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.6f} seconds")

def get_value(new_image):
    
    for image in [
        ("1.png", 1),
        ("2.png", 2),
        ("3.png", 3),
        ("4.png", 4),
        ("5.png", 5),
        ("none.png", 0),
        ("unknown.png", 9),
        ("flag.png", -1),
    ]:
        template = cv2.imread(f"resources/{image[0]}")
        image_umat = cv2.UMat(new_image)
        template_umat = cv2.UMat(template)
        # Perform template matching
        result = cv2.matchTemplate(image_umat, template_umat, cv2.TM_CCOEFF_NORMED)

        # Get the max value of the result to determine the best match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # If the max value is above the threshold, it's considered a match
        threshold = 0.9
        if max_val >= threshold:
            return image[1]

directions = [
    [-1,-1],[-1,0],[-1,1],
    [0,-1],[0,1],
    [1,-1],[1,0],[1,1]              
]

def mark_flags_in_grid():
    # print(grid)
    for i in range(1,15):
        for j in range(1,15):
            value = grid[i][j]
            print(f'Checking row: {i} col: {j} v: {value}')

            small_grid = [
                [grid[i-1][j-1],    grid[i-1][j],   grid[i-1][j+1]],
                [grid[i][j-1],      grid[i][j],     grid[i][j+1]],
                [grid[i+1][j-1],    grid[i+1][j],   grid[i+1][j+1]],
            ]
            flat = []
            flat.extend(small_grid[0])
            flat.extend(small_grid[1])
            flat.extend(small_grid[2])
            number_unknowns = flat.count(9)
            number_unknowns_flags = flat.count(-1)
            no_unknowns = number_unknowns_flags + number_unknowns
            
            if value in [1,2,3,4,5] and value == no_unknowns:
                for direction in directions:
                    row = i + direction[0]
                    col = j + direction[1]
                    if grid[row][col] == 9:
                        grid[row][col] = -1
                        print(small_grid)
                        right_click(col, row)
                        
def select_safe_in_grid():
    # print(grid)
    for i in range(1,15):
        for j in range(1,15):
            # if n is center, n is unknown, and the rest are not unknown, mark unknown as bomb
            value = grid[i][j]
            print(f'Checking row: {i} col: {j} v: {value}')

            small_grid = [
                [grid[i-1][j-1],    grid[i-1][j],   grid[i-1][j+1]],
                [grid[i][j-1],      grid[i][j],     grid[i][j+1]],
                [grid[i+1][j-1],    grid[i+1][j],   grid[i+1][j+1]],
            ]
            flat = []
            flat.extend(small_grid[0])
            flat.extend(small_grid[1])
            flat.extend(small_grid[2])
            number_unknowns_flags = flat.count(-1)
            unknowns = flat.count(9)
            
            if value in [1,2,3,4,5] and value == number_unknowns_flags and unknowns:
                print(f'flip this as center: {i} col: {j} v: {value}')
                for direction in directions:
                    row = i + direction[0]
                    col = j + direction[1]
                    if grid[row][col] == 9:
                        # grid[row][col] = -1
                        # print(small_grid)
                        left_click(col, row)
            # elif value == 9 and unknowns and number_unknowns_flags == :
            #     print(f'flip this as center: {i} col: {j} v: {value}')
            #     for direction in directions:
            #         row = i + direction[0]
            #         col = j + direction[1]
            #         if grid[row][col] == 9:
            #             # grid[row][col] = -1
            #             # print(small_grid)
            #             left_click(col, row)
while actions_taken != 0:
    actions_taken = 0
    read_grid()
    mark_flags_in_grid()
    select_safe_in_grid()