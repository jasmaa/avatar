import cv2
import dlib
from imutils import face_utils
from PIL import Image
import numpy as np

def find_coeffs(pa, pb):
    """Perspective transformation from planes
    https://stackoverflow.com/questions/14177744/how-does-perspective-transformation-work-in-pil
    """
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.matrix(matrix, dtype=np.float)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)



detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')

cam = cv2.VideoCapture(0)
cv2.namedWindow("test", cv2.WINDOW_GUI_NORMAL)

shades_im = Image.open('shades.png')
placement = {'x':0, 'y':0}

while True:
    ret, frame = cam.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    """
    # Draw landmarks
    for rect in rects:
        (x0, y0, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (x0, y0), (x0+w, y0+h), (0, 255, 0))

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        for (i, (x, y)) in enumerate(shape):
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
            cv2.putText(frame, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    """

    # record one position for now
    if rects:
        rect = rects[0]

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        placement['x'] = shape[27][0]
        placement['y'] = shape[27][1]


        pa = [
            [0, 0],
            [shades_im.size[0], 0],
            [shades_im.size[0], shades_im.size[1]],
            [0, shades_im.size[1]],
        ]

        u_diff = (1.5*(shape[45]-shape[36])).astype(int)
        v_diff = (1.5*(shape[30]-shape[27])).astype(int)
        v_0 = (0.3*(shape[30]-shape[27])).astype(int)
        pb = [
            shape[27] - u_diff - v_diff - v_0,
            shape[27] + u_diff - v_diff - v_0,
            shape[27] + u_diff + v_diff - v_0,
            shape[27] - u_diff + v_diff - v_0,
        ]

        """
        for (i, (x, y)) in enumerate(pb):
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
        for (i, (x, y)) in enumerate(pa):
            cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)
        """

        # PIL processing
        frame_im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        
        coeffs = find_coeffs(pb, pa)
        shades_im_mod = shades_im.transform(frame_im.size, Image.PERSPECTIVE, coeffs, Image.BICUBIC)
        frame_im.paste(
            shades_im_mod,
            (0, 0),
            shades_im_mod,
        )

        frame = cv2.cvtColor(np.array(frame_im), cv2.COLOR_RGB2BGR)

    cv2.imshow("test", frame)
    
    if not ret:
        break
    k = cv2.waitKey(1)

    if k == ord('q'):
        # ESC pressed
        print("Quitting...")
        break

cam.release()
cv2.destroyAllWindows()
