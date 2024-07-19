import numpy as np
import cv2 as cv
import os
import glob

input_dir = 'emergency\\test'
if input_dir.endswith('/'):
    input_dir = input_dir[:-1]
if os.path.isfile(input_dir):
    img_list = [input_dir]
else:
    img_list = sorted(glob.glob(os.path.join(input_dir, '*')))

def show_rgb_equalized(image, basename):
    channels = cv.split(image)
    eq_channels = []
    for ch, color in zip(channels, ['B', 'G', 'R']):
        eq_channels.append(cv.equalizeHist(ch))

    eq_image = cv.merge(eq_channels)
    eq_image = cv.cvtColor(eq_image, cv.COLOR_BGR2RGB)
    cv.imwrite(f'{basename}_EH.jpg',eq_image)

def histogram_equalization():
    for imagepath in img_list:
        print(imagepath)
        img_name = os.path.basename(imagepath)
        basename, ext = os.path.splitext(img_name)

        img = cv.imread(imagepath, basename)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        assert img is not None, "file could not be read, check with os.path.exists()"
        
        show_rgb_equalized(img)

def main():
    histogram_equalization()


if __name__ == '__main__':
    main()