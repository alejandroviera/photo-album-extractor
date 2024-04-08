import os
import cv2
import numpy as np

def read_image(filename):
    image = None
    if os.path.exists(filename) and os.path.isfile(filename):
        image = cv2.imread(filename)
        if image is None:
            with open(filename, 'rb') as binary_file:
                image_bytes = bytearray(binary_file.read())
    
            image = cv2.imdecode(np.asarray(image_bytes, dtype=np.uint8), -1)
    
    return image


def write_image(filename, image, jpg_quality = 100):
    result = False
    extension = os.path.splitext(os.path.basename(filename))[1]
    args = None
    if extension == ".jpg" and not jpg_quality is None:
        args = [cv2.IMWRITE_JPEG_QUALITY, jpg_quality]
    
    h, w = image.shape[0:2]

    if h > 0 and w > 0:
        result = cv2.imwrite(filename, image)
        if not result:
            _, image_bytes = cv2.imencode(extension, image, args)
            with open(filename, "wb") as write_file:
                result = write_file.write(image_bytes)
    
    return result