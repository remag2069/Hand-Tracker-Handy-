import cv2
import numpy as np
import math
from scipy import ndimage
import matplotlib.pyplot as plt
import pyautogui 


def detect_face(frmae):
    img_gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascPath = "suppliments/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    faces = faceCascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(110, 110)
        #flags = cv2.CV_HAAR_SCALE_IMAGE
    )
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),-2)
    
    return frame

def gen_mask(frame):
    lower_ycrcb = np.array([0,135,85], dtype=np.uint8)
    upper_ycrcb = np.array([255,180,135], dtype=np.uint8)     
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
    mask_ycrcb = cv2.morphologyEx(mask_ycrcb, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    global_mask=cv2.medianBlur(mask_ycrcb,3)
    global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))
    cv2.imshow('ycrcb',mask_ycrcb)

    return global_mask

def gen_contour(thresh,frame):
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hull = []
    for i in range(len(contours)):
        # creating convex hull object for each contour
        hull.append(cv2.convexHull(contours[i], False))
    # create an empty black image
    drawing_all = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)
    drawing_max = frame
    # draw contours and hull points
    for i in range(len(contours)):
        color_contours = (0, 255, 0) # green - color for contours
        color = (255, 0, 0) # blue - color for convex hull
        # draw ith contour
        cv2.drawContours(drawing_all, contours, i, color_contours, 1, 8, hierarchy)
        # draw ith convex hull object
        cv2.drawContours(drawing_all, hull, i, color, 1, 8)

    color_contours = (0, 0, 255) # green - color for contours
    color = (255, 0, 0) # blue - color for convex hull
    # draw ith contour
    cnt = max(contours, key = cv2.contourArea)
    epsilon = 0.0005*cv2.arcLength(cnt,True)
    approx= cv2.approxPolyDP(cnt,epsilon,True)

    hull = cv2.convexHull(cnt)
    hull = cv2.convexHull(approx, returnPoints=False)
    defects = cv2.convexityDefects(approx, hull)
    # print(defects,"*********")
    l=0
    sum_x=0
    sum_y=0
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(approx[s][0])
        end = tuple(approx[e][0])
        far = tuple(approx[f][0])
        pt= (100,180)
        
        
        # find length of all sides of triangle
        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
        s = (a+b+c)/2
        ar = math.sqrt(s*(s-a)*(s-b)*(s-c))
        
        #distance between point and convex hull
        d=(2*ar)/a
        
        # apply cosine rule here
        angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
        
    
        # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
        if angle <= 90 and d>30:
            l += 1
            cv2.circle(drawing_max, far, 3, [255,255,0], -1)
            sum_x+=far[0]
            sum_y+=far[1]

        #draw lines around hand
        cv2.line(drawing_max,start, end, [0,255,0], 2)
        cv2.circle(drawing_max, (sum_x,sum_y), 3, [0,0,255], -1)
    # cv2.drawContours(drawing_max, cnt, 0, color_contours, 1, 8)
        # draw ith convex hull object
    # cv2.drawContours(drawing_max, hull, 0, color, 1, 8)

    x,y,w,h = cv2.boundingRect(cnt)

    # draw the biggest contour (c) in green
    cv2.rectangle(drawing_max,(x,y),(x+w,y+h),(0,0,255),2)

    cv2.imshow('d',drawing_all)
    cv2.imshow('dm',drawing_max)


cap = cv2.VideoCapture(0)
     


while(1):
        
    # try:  #an error comes if it does not find anything in window as it cannot find contour of max area
      #therefore this try error statement
      
    ret, frame = cap.read()
    frame=cv2.flip(frame,1)
    # frame=cv2.resize(frame,(640,360))
    
    #define region of interest
    # roi=frame[100:300, 100:300]
    
    frame=detect_face(frame)

    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # d_t=ndimage.distance_transform_edt(global_mask)
    
    global_mask=gen_mask(frame)
    gen_contour(global_mask,frame)
    # d_t=cv2.distanceTransform(global_mask, cv2.DIST_L2,3)
    # max_val=np.max(d_t)
    # pos=np.where(d_t==max_val)
    # print(pos[0],',',pos[1],'*******',len(pos[1]),max_val)
    
    # cv2.circle(frame,(pos[1],pos[0]),10,(0,0,255),-2)
    
    # x=pos[1]*1920/640
    # y=pos[0]*1080/360
    
    # pyautogui.moveTo(x,y) 
    
    cv2.imshow('frame',frame)
        
    # except:
    #     pass

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