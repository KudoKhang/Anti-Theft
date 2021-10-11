import cv2
import datetime
import imutils
import pyglet


cap = cv2.VideoCapture('video.mp4')

top_left, bottom_right = (400, 350), (1000, 700)

while True:
    _, frame = cap.read()
    if frame is None:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    first_frame = frame
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    first_gray = cv2.GaussianBlur(first_gray, (21, 21), 0)
    
    frameDelta = cv2.absdiff(first_gray, gray)

    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
    for c in cnts:
        if cv2.contourArea(c) < 300:
            continue

        (x, y, w, h) = cv2.boundingRect(c)

        cx = x + w/2
        cy = y + h/2

        logic = top_left[0] < cx < bottom_right[0] and top_left[1] < cy < bottom_right[0]
        if logic:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            text = "Warning!!!"
            cv2.putText(frame, text, (500, 200), cv2.FONT_HERSHEY_SIMPLEX, 2,  (0, 0, 255), 2)
            
            ## Capture camera
            # time = datetime.datetime.now()
            # cv2.imwrite(f'Records/{time}.jpg', frame)

            ## Record video
            # w = cap.get(cv2.CAP_PROP_FRAME_WIDTH);
            # h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT); 
            # out = cv2.VideoWriter('Records/out.avi',cv2.VideoWriter_fourcc(*'XVID'), 20, (int(w),int(h)))
            
            ## Play sound warning
            # music = pyglet.resource.media('alert.wav')
            # music.play()
    
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
