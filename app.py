import cv2
import dlib
from imutils import face_utils
from PIL import Image
import numpy as np

import rendering

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')

cam = cv2.VideoCapture(0)
cv2.namedWindow("test", cv2.WINDOW_GUI_NORMAL)

avatar = rendering.Avatar()
avatar.load('data/shades')

while True:
    ret, frame = cam.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    
    # TEMP: Draw landmarks
    for rect in rects:
        (x0, y0, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (x0, y0), (x0+w, y0+h), (0, 255, 0))

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        for (i, (x, y)) in enumerate(shape):
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
            cv2.putText(frame, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    frame_im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    #frame_im = Image.new('RGBA', frame_im.size, color='white') # TEMP: white out for cartoon

    # Render for all faces
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        frame_im = avatar.render(frame_im, shape)
        
    # Display frame
    frame = cv2.cvtColor(np.array(frame_im), cv2.COLOR_RGB2BGR)
    cv2.imshow("test", frame)
    
    if not ret:
        break

    k = cv2.waitKey(1)

    if k == ord('q'):
        print("Quitting...")
        break

cam.release()
cv2.destroyAllWindows()
