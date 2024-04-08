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

points = []

#cropping
rectangle_thickness = 2
cropChanging = False
topLeft = (0,0)
bottomRight = (0, 0)
original_image_cropped = None

def create_argparser():
    parser = argparse.ArgumentParser("Extracts each photo detected in scanned photo album pages")
    parser.add_argument('--input-folder', '-i', help='Path that contains the images of all the scanned photo album pages.', required=False)
    parser.add_argument('--debug', '-d', help='Indicates if you want to generate output images showing the results of each step of the process.', default=False, action=argparse.BooleanOptionalAction, type=bool)
    parser.add_argument('--verbose', '-v', help='Indicates if you want a verbose output in the console.', default=False, action=argparse.BooleanOptionalAction, type=bool)

    return parser


def save(output_folder, base_file_name, index):
    if len(points) < 4:
        return False
    
    sorted_points = sorted(points, key = lambda x: x[0] + x[1])
    # We want to make sure we return the points in this order: top-left, bottom-left, top-right, bottom-right
        # This assumption is used when the homography is applied to warp the perspective

    if sorted_points[2][0] < sorted_points[1][0]:
        sorted_points[1], sorted_points[2] = sorted_points[2], sorted_points[1]

    sorted_points = np.array(sorted_points)
    sorted_points = np.round(sorted_points / resizeRatio).astype(np.int32)
    x,y,w,h = cv2.boundingRect(sorted_points)
    new_points = np.array([
        [0, 0],
        [0, h],
        [w, 0],
        [w, h]
        ])
    homography, _ = cv2.findHomography(sorted_points, new_points, cv2.RANSAC)

    photo = cv2.warpPerspective(original_image_cropped, homography, (w, h))

    output_file = os.path.join(output_folder, base_file_name + "-" + str(index) + ".jpg")
    write_image(output_file, photo)

    return True


def crop():
    global original_image_cropped
    if bottomRight[0] == topLeft[0] or bottomRight[1] == topLeft[1]:
        return False
    
    h, w = working_image.shape[0:2]
    
    minX = max(min(topLeft[0], bottomRight[0]), 0) # The MAX function is used to be protected against a negative X value
    maxX = min(max(topLeft[0], bottomRight[0]), w)
    minY = max(min(topLeft[1], bottomRight[1]), 0) # The MAX function is used to be protected against a negative X value
    maxY = min(max(topLeft[1], bottomRight[1]), h)

    if minX != maxX and minY != maxY:
        original_image_cropped = original_image_cropped[int(minY / resizeRatio) : int(maxY / resizeRatio), int(minX / resizeRatio) : int(maxX / resizeRatio)]
        show_image()


def reset():
    global points, original_image_cropped
    points.clear()
    original_image_cropped = original_image.copy()


def show_image():
    global working_image, resizeRatio
    working_image = original_image_cropped.copy()
    
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

    cv2.imshow(window_name, working_image)

    return True


def cropAreaChanged(action, x, y, flags, userdata):
    global cropChanging, topLeft, bottomRight
    if action == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            cv2.circle(working_image, (x, y), 2, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow(window_name, working_image)
    elif action == cv2.EVENT_RBUTTONDOWN:
        topLeft = (x,y)
        cropChanging = True
    elif action == cv2.EVENT_RBUTTONUP:
        cropChanging = False
        crop()
    elif action == cv2.EVENT_MOUSEMOVE and cropChanging:
        bottomRight = (x, y)
        tempFrame2 = working_image.copy()
        cv2.rectangle(tempFrame2, topLeft, bottomRight, (0,255,0), rectangle_thickness, cv2.LINE_AA)
        cv2.imshow(window_name, tempFrame2)


def process_file(input_file, output_folder):
    global original_image, original_image_cropped, working_image, points
    
    base_file_name = os.path.splitext(os.path.basename(input_file))[0]
    original_image = read_image(input_file)
    
    reset()
    show_image()
    
    exit = False
    image_index = 1

    while True:
        k = cv2.waitKey(0)
        if k == ord('q'):
            exit = True
            break
        elif k == ord('n'):
            break
        elif k == ord('h'):
            working_image = imutils.rotate_bound(working_image, 90)
            original_image = imutils.rotate_bound(original_image, 90)
            original_image_cropped = imutils.rotate_bound(original_image_cropped, 90)
            points.clear()
            show_image()
        elif k == ord('g'):
            working_image = imutils.rotate_bound(working_image, -90)
            original_image = imutils.rotate_bound(original_image, -90)
            original_image_cropped = imutils.rotate_bound(original_image_cropped, -90)
            points.clear()
            show_image()
        elif k == ord('s'):
            save(output_folder, base_file_name, image_index)
            image_index += 1
            reset()
            show_image()
        elif k == ord('r'):
            reset()
            show_image()

    return exit


def main():
    global debug, monitor_resolution

    args = create_argparser().parse_args()

    if args.input_folder is None:
        input_folder = easygui.diropenbox(title="Select a folder")
        if not input_folder:
            return
    
    output_folder = os.path.join(input_folder, 'crops')
    debug = args.debug

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