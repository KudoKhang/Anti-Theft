import cv2
import numpy as np
from playsound import playsound
from pygame import mixer

mixer.init()
mixer.music.load('./Sources/alert.wav')


backSub = cv2.createBackgroundSubtractorKNN()
# backSub = cv2.createBackgroundSubtractorMOG2()
# Nhận xét: KNN tốt hơn MOG2

top_left, bottom_right = (400, 200), (1000, 700) # ROI

cap = cv2.VideoCapture('./Sources/video.mp4')

while True:
    _, frame = cap.read()

    fgMask = backSub.apply(frame)
    fgMask = cv2.cvtColor(fgMask, 0) # 0: grayScale

    # Khu nhieu
    kernel = np.ones((5,5), np.uint8)
    fgMask = cv2.erode(fgMask, kernel, iterations=1) 
    fgMask = cv2.dilate(fgMask, kernel, iterations=1)
    fgMask = cv2.GaussianBlur(fgMask, (3,3), 0)
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)
    _,fgMask = cv2.threshold(fgMask,130,255,cv2.THRESH_BINARY)

    fgMask = cv2.Canny(fgMask, 20, 200)
    contours, _ = cv2.findContours(fgMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.rectangle(frame, top_left, bottom_right, (255, 255, 0), 2)

    for i in range(len(contours)):
        (x, y, w, h) = cv2.boundingRect(contours[i])
        cx = x + w/2
        cy = y + h/2

        logic = top_left[0] < cx < bottom_right[0] and top_left[1] < cy < bottom_right[1]
        area = cv2.contourArea(contours[i])
        if area < 500:
            continue
    
        if logic:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Warning", (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            mixer.music.play()

            # playsound('./Sources/alert.wav', block=False) # Crashhh ????
    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cv2.destroyAllWindows()

