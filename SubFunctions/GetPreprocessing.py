from dataclasses import dataclass
import cv2
import numpy as np
from termcolor import cprint


@dataclass
class Preprocessing(object):
    # Initialize with the input image
    def __init__(self, image):
        self.image = image

    # Static method for extracting the Region of Interest (ROI) from the image
    @staticmethod
    def roi_extraction(img):
        # Convert to grayscale
        gray = img.copy().astype(np.uint8)

        # Threshold: non-black pixels are set to 255
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get bounding box of the largest contour
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)

            # Crop the image to the bounding box
            cropped_img = img[y:y + h, x:x + w]

            return cropped_img
        else:
            print("No non-black region found!")
            return img

    # Method to apply preprocessing and ROI extraction sequentially
    def get_preprocessing(self):
        cprint('\n')
        cprint("================================", color='blue')
        cprint("[⚠️] Preprocessing ", color='grey', on_color='on_yellow')
        cprint("================================", color='blue')

        # Apply ROI extraction on the preprocessed image
        roi = self.roi_extraction(self.image)
        roi = cv2.resize(roi, (96, 96))

        # Return the cropped region of interest (ROI)
        return roi
