import numpy as np
import cv2 as cv
import os
import glob

class ImageProcessing():
    def __init__(self, 
                 input='emergency\\test',
                 output='results_imageprocessing'):
        self.input=input
        if self.input.endswith('/') or self.input.endswith('\\'):
            self.input = self.input[:-1]
        if os.path.isfile(self.input):
            self.input = [self.input]
        else:
            self.input = sorted(glob.glob(os.path.join(self.input, '*')))

    def show_rgb_equalized(self, image, basename):
        channels = cv.split(image)
        eq_channels = []
        for ch, color in zip(channels, ['B', 'G', 'R']):
            eq_channels.append(cv.equalizeHist(ch))

        eq_image = cv.merge(eq_channels)
        eq_image = cv.cvtColor(eq_image, cv.COLOR_BGR2RGB)
        cv.imwrite(f'{basename}_EH.jpg',eq_image)

    def histogram_equalization(self):
        for imagepath in self.input:
            print(imagepath)
            img_name = os.path.basename(imagepath)
            basename, ext = os.path.splitext(img_name)

            img = cv.imread(imagepath, basename)
            img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
            assert img is not None, "file could not be read, check with os.path.exists()"
            
            self.show_rgb_equalized(img, basename)

def main():
    IP = ImageProcessing(input='emergency\\test')
    IP.histogram_equalization()


if __name__ == '__main__':
    main()