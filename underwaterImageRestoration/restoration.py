import numpy as np
from helper import *
import datetime
import cv2
from GuidedFilter import GuidedFilter
from copy import deepcopy

class Restoration():
    def __init__(self, img, windowSize = 15):
        self.input = img
        self.windowSize = windowSize
        self.normI = (img - img.min()) / (img.max() - img.min())
        self.D = self.determineDepth()
        starttime = datetime.datetime.now()
        self.backgroundlight = self.getBackgroundLight()
        Endtime = datetime.datetime.now()
        print('Time', Endtime - starttime)
        # starttime = datetime.datetime.now()
        # self.backgroundlight2 = self.getAtomsphericLight()
        # Endtime = datetime.datetime.now()
        # print('Time', Endtime - starttime)
        self.transmission = self.getTransmission()
        self.refinedtransmission = self.getRefinedtransmission()
        # self.refinedtransmission = cv2.equalizeHist(np.uint8(self.transmission[:, :, 0] * 255))
        self.normJ_blue, self.normJ_green = self.dehazed_BG()
        self.norm_restored = self.RC_correction()
        self.output = self.adaptiveExp_map()

    def getRefinedTransmission(self):
        return 1
        

    def determineDepth(self):
        img , blockSize = self.normI, self.windowSize
        img_GB = getMAxBlueGreenChannel(img, type = 'BGR')
        Max_GB = getDarkChannel(img_GB, blockSize)
        Max_R  = getDarkChannel(img[:,:,2], blockSize)
        largestDiff = Max_R  - Max_GB
        return largestDiff   

    def getBackgroundLight(self):
        M, N, C = self.normI.shape #M are the rows, N are the columns, C is the bgr channel
        flatD = self.D.reshape(M*N)  
        flatI = self.normI.reshape(M*N,3)
        searchidx = flatD.argsort(axis=0)[:1]
        searchidx = searchidx.ravel()
        return np.average(flatI.take(searchidx, axis=0), axis = 0)

    def getTransmission(self):
        M, N, C = self.normI.shape #M are the rows, N are the columns, C is the bgr channel
        B, w = self.backgroundlight, self.windowSize
        padwidth = math.floor(w/2)
        padded = np.pad(self.normI/B, ((padwidth, padwidth), (padwidth, padwidth),(0,0)), 'constant')
        transmission = np.zeros((M,N,2))
        for y, x in np.ndindex(M, N):
            transmission[y,x,0] = 1 - np.min(padded[y : y+w , x : x+w , 0])
            transmission[y,x,1] = 1 - np.min(padded[y : y+w , x : x+w , 1])
        return transmission

    def getRefinedtransmission(self):
        T = deepcopy(self.transmission)
        img =  self.input
        gimfiltR = 50  
        eps = 10 ** -3  

        guided_filter = GuidedFilter(img, gimfiltR, eps)
        T[:,:,0] = guided_filter.filter(T[:,:,0])
        T[:,:,1] = guided_filter.filter(T[:,:,1])
        T = np.clip(T, 0.1, 0.9)

        return T

    def dehazed_BG(self):
        B = self.backgroundlight
        # print(self.normI.shape)
        # print(self.refinedtransmission.shape)
        refinedt_blue, refinedt_green = self.refinedtransmission[:,:,0] /255, self.refinedtransmission[:,:,1] /255
        J_blue = (self.normI[:,:,0] - B[0])/refinedt_blue + B[0]
        normJ_blue = (J_blue - J_blue.min()) / (J_blue.max() - J_blue.min())
        J_green = (self.normI[:,:,1] - B[1])/refinedt_green + B[1]
        normJ_green = (J_green - J_green.min()) / (J_green.max() - J_green.min())
        return normJ_blue, normJ_green

    def RC_correction(self):
        normJ_blue,normJ_green = self.normJ_blue, self.normJ_green
        avgRr = 1.5 - np.average(normJ_blue.ravel()) - np.average(normJ_green.ravel())
        compCoeff = avgRr/np.average(self.normI[:,:,2].ravel())
        Rrec = self.normI[:,:,2]*compCoeff
        normRrec = (Rrec - Rrec.min()) / (Rrec.max() - Rrec.min())    
        restored = np.zeros(np.shape(self.normI))
        restored[:,:,0] = normJ_blue; 
        restored[:,:,1] = normJ_green; 
        restored[:,:,2] = normRrec; 
        return restored

    def AdaptiveExposureMap(img, sceneRadiance, Lambda, blockSize):

        minValue = 10 ** -2
        img = np.uint8(img)
        sceneRadiance = np.uint8(sceneRadiance)

        YjCrCb = cv2.cvtColor(sceneRadiance, cv2.COLOR_BGR2YCrCb)
        YiCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        normYjCrCb = (YjCrCb - YjCrCb.min()) / (YjCrCb.max() - YjCrCb.min())
        normYiCrCb = (YiCrCb - YiCrCb.min()) / (YiCrCb.max() - YiCrCb.min())
        Yi = normYiCrCb[:, :, 0]
        Yj = normYjCrCb[:, :, 0]
        Yi = np.clip(Yi, minValue,1)
        Yj = np.clip(Yj, minValue,1)

        # print('np.min(Yi)',np.min(Yi))
        # print('np.max(Yi)',np.max(Yi))
        # print('np.min(Yj)',np.min(Yj))
        # print('np.max(Yj)',np.max(Yj))
        # Yi = YiCrCb[:, :, 0]
        # Yj = YjCrCb[:, :, 0]
        S = (Yj * Yi + 0.3 * Yi ** 2) / (Yj ** 2 + 0.3 * Yi ** 2)

        # print('S',S)

        gimfiltR = 50  # 引导滤波时半径的大小
        eps = 10 ** -3  # 引导滤波时epsilon的值

        # refinedS = guided_filter_he(YiCrCb, S, gimfiltR, eps)

        guided_filter = GuidedFilter(YiCrCb, gimfiltR, eps)
        # guided_filter = GuidedFilter(normYiCrCb, gimfiltR, eps)

        refinedS = guided_filter.filter(S)

        # print('guided_filter_he(YiCrCb, S, gimfiltR, eps)', refinedS)
        # S = np.clip(S, 0, 1)

        # cv2.imwrite('OutputImages_D/' + 'SSSSS' + '_GBdehazingRCorrectionStretching.jpg', np.uint8(S * 255))

        S_three = np.zeros(img.shape)
        S_three[:, :, 0] = S_three[:, :, 1] = S_three[:, :, 2] = refinedS

        return S_three

    def adaptiveExp_map(self):   
        r= 50
        eps= 10 ** -2
        restored = deepcopy(self.norm_restored)
        R = (restored*255).astype(np.uint8)
        I = (self.normI*255).astype(np.uint8)
        YjCrCb = cv2.cvtColor(R, cv2.COLOR_BGR2YCrCb)  
        YiCrCb = cv2.cvtColor(I, cv2.COLOR_BGR2YCrCb)  
        normYjCrCb = (YjCrCb - YjCrCb.min())/(YjCrCb.max() - YjCrCb.min())
        normYiCrCb = (YiCrCb - YiCrCb.min())/(YiCrCb.max() - YiCrCb.min())
        Yi = normYiCrCb[:,:,0]
        Yj = normYjCrCb[:,:,0]
        S = (Yj*Yi + 0.3*Yi**2)/(Yj**2 + 0.3*Yi**2)
        # refinedS = guided_filter(normYiCrCb, S, r, eps) 
        guided_filter = GuidedFilter(YiCrCb, r, eps)
        refinedS = guided_filter.filter(S)
        M,N = S.shape
        rs = np.zeros((M,N,3))
        rs[:,:,0] = rs[:,:,1] = rs[:,:,2] = refinedS 
        OutputExp = restored*rs
        return (OutputExp - OutputExp.min())/(OutputExp.max() - OutputExp.min())


if __name__ == '__main__':
    index = '7'
    img = cv2.imread('img/'+index+'.jpg') 
    scale_percent = 100 # percent of original size
    dim0 = (img.shape[1], img.shape[0])
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    I = resized
    x = Restoration(img = resized)
    print(x.backgroundlight)
    print('--------------')
    # print(x.backgroundlight2[2])
    cv2.imwrite('result0/' +index+ '_transmission.jpg', np.uint8(x.transmission[:, :, 0] * 255))
    cv2.imwrite('result0/'  +index+ '_refinedtransmission.jpg', np.uint8(x.refinedtransmission[:, :, 0] * 255))
    # cv2.imwrite('result/' + 'refinedtransmission.jpg', np.uint8(x.refinedtransmission))
    cv2.imwrite('result0/'  +index+ '_output.jpg', np.uint8(cv2.resize(x.output * 255, dim0, interpolation = cv2.INTER_AREA)))
    # cv2.imwrite('result/' + 'output.jpg', x.output * 255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
