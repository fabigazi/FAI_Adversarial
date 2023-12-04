import os
import cv2
import numpy as np
from PIL import Image

num_evolutions = 1000
# convert this numpy array into Image
im = cv2.imread(f"./genetic/untargetted/10.png")
width, height, layers = im.shape
# print(width, height)
print(im.shape)
imez = Image.fromarray(im)
imez.show()
# im.show()
# fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter("./genetic_untargetted.avi", 0, 1, (width, height))

for i in range(10, num_evolutions + 1, 10):
    im = cv2.imread(f"./genetic/untargetted/{i}.png")
    video.write(im)

cv2.destroyAllWindows()
video.release()
