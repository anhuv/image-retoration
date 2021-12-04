import cv2
import math
import numpy as np

def getDarkChannel(img,blockSize = 3):

    if len(img.shape)==2:
        pass
    else:
        print("bad image shape, input image must be two demensions")
        return None

    if blockSize % 2 == 0 or blockSize < 3:
        print('blockSize is not odd or too small')
        return None

    A = int((blockSize-1)/2) #AddSize

    #New height and new width
    H = img.shape[0] + blockSize - 1
    W = img.shape[1] + blockSize - 1

    # imgMiddle
    imgMiddle = 255 * np.zeros((H,W))    

    imgMiddle[A:H-A, A:W-A] = img
    
    imgDark = np.zeros_like(img, np.float16)    

    for i in range(A, H-A):
        for j in range(A, W-A):
            x = range(i-A, i+A+1)
            y = range(j-A, j+A+1)
            imgDark[i-A,j-A] = np.max(imgMiddle[x,y])                            
            
    return imgDark

def Background_light(normI,w=100):
    M, N, C = normI.shape #M are the rows, N are the columns, C is the bgr channel
    padwidth = math.floor(w/2)
    padded = np.pad(normI, ((padwidth, padwidth), (padwidth, padwidth),(0,0)), 'constant')
    D = np.zeros((M,N,2))
    for y, x in np.ndindex(M, N):
        D[y,x,0] = np.amax(padded[y : y+w , x : x+w , 2]) - np.amax(padded[y : y+w , x : x+w , 0])
        D[y,x,1] = np.amax(padded[y : y+w , x : x+w , 2]) - np.amax(padded[y : y+w , x : x+w , 1])
        # break
    flatD = D.reshape(M*N,2)  
    flatI = normI.reshape(M*N,3)
    searchidx = flatD.argsort(axis=0)[:1]
    searchidx = searchidx.ravel()
    return np.average(flatI.take(searchidx, axis=0), axis = 0)

def getMAxBlueGreenChannel(img, type = 'RGB'):
    if type == 'BGR':
        return np.amax(img[:,:,[0,1]], axis=2)
    else:
        return np.amax(img[:,:,[1,2]], axis=2)
