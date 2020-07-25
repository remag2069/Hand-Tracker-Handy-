import cv2
import numpy as np
import math
from scipy import ndimage
import matplotlib.pyplot as plt
import pyautogui 


def detect_face(img, faceCascade):
    faces = faceCascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(110, 110)
        #flags = cv2.CV_HAAR_SCALE_IMAGE
    )
    return faces


cap = cv2.VideoCapture(0)
     
while(1):
        
    try:  #an error comes if it does not find anything in window as it cannot find contour of max area
      #therefore this try error statement
      
        ret, frame = cap.read()
        frame=cv2.flip(frame,1)
        frame=cv2.resize(frame,(640,360))
        
        #define region of interest
        # roi=frame[100:300, 100:300]
        
        lower_hsv = np.array([0,15,0], dtype=np.uint8)
        upper_hsv = np.array([17,170,255], dtype=np.uint8)

        lower_ycrcb = np.array([0,135,85], dtype=np.uint8)
        upper_ycrcb = np.array([255,180,135], dtype=np.uint8)
        
     #extract skin colur imagw  
        img_gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cascPath = "suppliments/haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(cascPath)
        faces = detect_face(img_gray, faceCascade)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),-2) 

 
        # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        # yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        # hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

        # mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
        # mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

        mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
        mask_ycrcb = cv2.morphologyEx(mask_ycrcb, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

        # global_mask=cv2.bitwise_and(mask_ycrcb,mask_hsv)
        global_mask=cv2.medianBlur(mask_ycrcb,3)
        global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))
        # d_t=ndimage.distance_transform_edt(global_mask)
        # global_mask=cv2.bitwise_not(global_mask)

        d_t=cv2.distanceTransform(mask_ycrcb, cv2.DIST_L2,3)
        # d2_5=cv2.distanceTransform(mask_ycrcb, cv2.DIST_L2,3)
        # d2_1=cv2.distanceTransform(mask_ycrcb, cv2.DIST_L2,3)
        # d1_3=cv2.distanceTransform(mask_ycrcb, cv2.DIST_L1,3)

        # cv2.imshow('hsv',hsv)
        # cv2.imshow('ycrcb',ycrcb)
        # cv2.imshow('d_t',d_t)
        # cv2.imshow('d2_3',cv2.bitwise_not(d2_3))
        # cv2.imshow('d2_5',cv2.bitwise_not(d2_5))
        # cv2.imshow('d2_1',cv2.bitwise_not(d2_1))
        # cv2.imshow('d1_3',cv2.bitwise_not(d1_3))
        # cv2.imshow('yuv',yuv)
        # cv2.imshow('hls',hls)
        # cv2.imshow('mask_hsv',mask_hsv)
        # cv2.imshow('mask_ycrcb',mask_ycrcb)
        # cv2.imshow('global_mask',global_mask)
        max_val=np.max(d_t)
        # print(max_val)
        pos=np.where(d_t==max_val)
        print(pos[0],',',pos[1],'*******',len(pos[1]),max_val)
        cv2.circle(frame,(pos[1],pos[0]),10,(0,0,255),-2)
        x=pos[1]*1920/640
        y=pos[0]*1080/360
        pyautogui.moveTo(x,y) 
        cv2.imshow('frame',frame)
        
    except:
        pass

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

# print(np.unique(d_t))
# print('*******************')
# print(np.unique(global_mask))
# print('*******************')
# print(np.unique(np.subtract(global_mask,d_t)))
print(len(frame),len(frame[0]))

cv2.destroyAllWindows()
cap.release()    
plt.imshow(d_t)
plt.show()