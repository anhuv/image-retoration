import cv2
import numpy as np
import imageRestorationFns as ir

image = cv2.imread ("low.png")
K = 0.1
blur_kernel = np.array (cv2.imread ("5.bmp", 0))
outImage = ir.weiner_filter (image, blur_kernel, K)
cv2.imshow("output", outImage)
cv2.waitKey (0)
