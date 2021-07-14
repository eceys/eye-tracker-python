import cv2

vid = cv2.VideoCapture("eye_motion.mp4")


while True:
    ret, frame = vid.read()
    if ret is False:
        break

    roi = frame[80:210,230:450] #crop video

    
    rows, cols, _ = roi.shape  #frame size

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 3, 255, cv2.THRESH_BINARY_INV) #invert black and white


    contours,_ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = lambda x : cv2.contourArea(x), reverse=True) #contours sorted bigger to small

    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        cv2.rectangle(roi, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.line(roi, (int(x + (w/2)), 0), (int(x+ (w/2)), rows), (0, 255, 0), 2) #x1, y1   x2,y2   colour
        cv2.line(roi, (0, int(y + (h/2))), (cols, int(y + (h/2))), (0, 255, 0), 2)
        break

    frame[80:210, 230:450] = roi
    cv2.imshow("frame", frame)

    if cv2.waitKey(80) & 0xFF == ord("q"): #quit video
        break

vid.release()
cv2.destroyAllWindows()