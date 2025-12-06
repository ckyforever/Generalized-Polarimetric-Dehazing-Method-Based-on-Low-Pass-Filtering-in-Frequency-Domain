import os

import cv2

if __name__ == "__main__":
    directory = os.path.dirname(__file__)
    image_path = os.path.join(directory, "images/image.jpg")
    image = cv2.imread(image_path)
    
    if image is None:
        print("Image not found")