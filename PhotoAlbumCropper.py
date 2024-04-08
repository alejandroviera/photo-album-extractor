import cv2
import numpy as np
import argparse
import easygui
import imutils
import os
import glob
from IOHelper import write_image, read_image
from MonitorHelper import get_monitor_resolution

debug = False
window_name = 'Manual Photo Extractor'

original_image = None
working_image = None

monitor_resolution = (0, 0)
resizeRatio = 1

rectangle_thickness = 2
cropChanging = False
topLeft = (0,0)
bottomRight = (0, 0)
rotationAngle = 0

def create_argparser():
    parser = argparse.ArgumentParser("Extracts each photo detected in scanned photo album pages")
    parser.add_argument('--input-folder', '-i', help='Path that contains the images of all the scanned photo album pages.', required=False)

    return parser


def save(output_folder, base_file_name):
    if bottomRight[0] == topLeft[0] or bottomRight[1] == topLeft[1]:
        return False
    
    h, w = working_image.shape[0:2]
    
    minX = max(min(topLeft[0], bottomRight[0]), 0) # The MAX function is used to be protected against a negative X value
    maxX = min(max(topLeft[0], bottomRight[0]), w)
    minY = max(min(topLeft[1], bottomRight[1]), 0) # The MAX function is used to be protected against a negative X value
    maxY = min(max(topLeft[1], bottomRight[1]), h)

    if minX != maxX and minY != maxY:
        if rotationAngle != 0:
            temp_image = imutils.rotate_bound(original_image, rotationAngle)
            roi = temp_image[int(minY / resizeRatio) : int(maxY / resizeRatio), int(minX / resizeRatio) : int(maxX / resizeRatio)]
        else:
            roi = original_image
        output_file = os.path.join(output_folder, base_file_name + ".jpg")
        return write_image(output_file, roi)
    else:
        return False


def show_image():
    global working_image, resizeRatio

    working_image = original_image.copy()

    if rotationAngle != 0:
        working_image = imutils.rotate_bound(working_image, rotationAngle)
    
    h, w = working_image.shape[0:2]
    if w > monitor_resolution[0] or h > monitor_resolution[1]:
        ratio_width = monitor_resolution[0] / w
        ratio_height = monitor_resolution[1] / h
        resizeRatio = min(ratio_width, ratio_height) * 0.9
        new_width = int(w * resizeRatio)
        new_height = int(h * resizeRatio)
        working_image = cv2.resize(working_image, (new_width, new_height))
    else:
        resizeRatio = 1

    tempFrame2 = working_image.copy()
    cv2.rectangle(tempFrame2, topLeft, bottomRight, (0,255,0), rectangle_thickness, cv2.LINE_AA)

    cv2.imshow(window_name, tempFrame2)

    return True


def cropAreaChanged(action, x, y, flags, userdata):
    global cropChanging, topLeft, bottomRight
    if action == cv2.EVENT_LBUTTONDOWN:
        topLeft = (x,y)
        cropChanging = True
    elif action == cv2.EVENT_LBUTTONUP:
        cropChanging = False
    elif action == cv2.EVENT_MOUSEMOVE and cropChanging:
        bottomRight = (x, y)
        tempFrame2 = working_image.copy()
        cv2.rectangle(tempFrame2, topLeft, bottomRight, (0,255,0), rectangle_thickness, cv2.LINE_AA)
        cv2.imshow(window_name, tempFrame2)


def process_file(input_file, output_folder):
    global original_image, rotationAngle
    
    base_file_name = os.path.splitext(os.path.basename(input_file))[0]
    original_image = read_image(input_file)
    
    show_image()
    
    exit = False

    while True:
        k = cv2.waitKey(0)
        if k == ord('q'):
            exit = True
            break
        elif k == ord('n'):
            break
        elif k == ord('h'):
            rotationAngle += 90
            show_image()
        elif k == ord('g'):
            rotationAngle -= 90
            show_image()
        elif k == ord('s'):
            result = save(output_folder, base_file_name)
            if result:
                break # only one save per file. Automatically go for next

    return exit


def main():
    global debug, monitor_resolution

    args = create_argparser().parse_args()

    if args.input_folder is None:
        input_folder = easygui.diropenbox(title="Select a folder")
        if not input_folder:
            return
    
    output_folder = os.path.join(input_folder, 'crops')

    monitor_resolution = get_monitor_resolution()
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, cropAreaChanged)

    os.makedirs(output_folder, exist_ok=True)

    input_files = glob.glob(os.path.join(input_folder, '*.*'))
    
    for filename in input_files:
        if os.path.isfile(filename):
            exit = process_file(filename, output_folder)
            if exit:
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

