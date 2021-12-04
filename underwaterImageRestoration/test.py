import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('result\GBDehazedRcoorectionUDCP_TM.jpg',0)
img2 = cv.imread('result\GBDehazedRcoorectionUDCP_TM1.jpg',0)
# plt.hist(img.ravel(),256,[0,256])
# # plt.hist(img2.ravel(),256,[0,256])
# plt.show()

# import Opencv
import cv2
  
# import Numpy
import numpy as np
  
# read a image using imread
img =  cv.imread('result\GBDehazedRcoorectionUDCP_TM.jpg',0)
  
# creating a Histograms Equalization
# of a image using cv2.equalizeHist()
equ = cv2.equalizeHist(img)
  
# stacking images side-by-side
res = np.hstack((img, equ))
  
# show image input vs output
cv2.imshow('output', res)
  
cv2.waitKey(0)
cv2.destroyAllWindows()