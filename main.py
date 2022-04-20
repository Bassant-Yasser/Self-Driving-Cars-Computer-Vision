import numpy as np
import matplotlib.image as mpimg
import cv2
from IPython.display import HTML, Video
from moviepy.editor import VideoFileClip
from Thresholding import *
from PerspectiveTransformation import *
from LaneLines import *
import matplotlib.pyplot as plt
from PIL import Image
import sys

class FindLaneLines:
    """ This class is for parameter tunning.

    Attributes:
        ...
    """
    def __init__(self):
        """ Init Application"""
        self.thresholding = Thresholding()
        self.transform = PerspectiveTransformation()
        self.lanelines = LaneLines()

    def forward(self, img):
        out_img = np.copy(img)
        img = self.transform.forward(img)
        thresholded_img = self.thresholding.forward(img)    
        green_rectangles, green_lanes = self.lanelines.forward(thresholded_img)  
        img = self.transform.backward(green_lanes)
        out_img = cv2.addWeighted(out_img, 1, img, 0.6, 0)
        out_img = self.lanelines.plot(out_img, thresholded_img, green_lanes, green_rectangles)
        return out_img

    def process_image(self, input_path, output_path):
        img = mpimg.imread(input_path)
        out_img = self.forward(img)
        mpimg.imsave(output_path, out_img)

    def process_video(self, input_path, output_path):
        clip = VideoFileClip(input_path)
        out_clip = clip.fl_image(self.forward)
        out_clip.write_videofile(output_path, audio=False)

def main():
    args = sys.argv[1:]
    if (len(sys.argv) != 4):
        sys.exit(2)

    input = args[1]
    output = args[2]

    findLaneLines = FindLaneLines()
    if args[0] == '--video':
        findLaneLines.process_video(input, output)
    else:
        findLaneLines.process_image(input, output)




if __name__ == "__main__":
    main()