import cv2
import numpy as np
import imageRestorationFns as ir

image = cv2.imread ("low.png")
K = 0.5
blur_kernel = np.array (cv2.imread ("1.bmp", 0))
outImage = ir.full_inverse_filter (image, blur_kernel)
cv2.imshow("output", outImage)
cv2.waitKey (0)
