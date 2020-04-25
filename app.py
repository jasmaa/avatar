import cv2
import dlib
from imutils import face_utils
from PIL import Image
import numpy as np

import rendering

def str2bool(v):
    """Boolean argument parser
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main(predictor_path, avatar_path, draw_landmarks, draw_video, draw_detected_only):
    """Avatar rendering
    """
    
    # Load predictor model and avatar
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    avatar = rendering.Avatar()
    avatar.load(avatar_path)

    cam = cv2.VideoCapture(0)
    cv2.namedWindow("test", cv2.WINDOW_GUI_NORMAL)

    while True:
        ret, frame = cam.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        frame_im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Blank video feed
        if not draw_video:
            frame_im = Image.new('RGBA', frame_im.size, color='white')

        # Render for all faces
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            frame_im = avatar.render(frame_im, shape)

        frame = cv2.cvtColor(np.array(frame_im), cv2.COLOR_RGB2BGR)

        # Draw landmarks
        if draw_landmarks:
            for rect in rects:
                (x0, y0, w, h) = face_utils.rect_to_bb(rect)
                cv2.rectangle(frame, (x0, y0), (x0+w, y0+h), (0, 255, 0))

                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                for (i, (x, y)) in enumerate(shape):
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                    cv2.putText(frame, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        # Display frame
        if draw_detected_only:
            if rects:
                cv2.imshow("test", frame)
        else:
            cv2.imshow("test", frame)
        
        if not ret:
            break

        k = cv2.waitKey(1)

        if k == ord('q'):
            print("Quitting...")
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Render cartoon avatars')
    
    parser.add_argument('--predictor', default='data/shape_predictor_68_face_landmarks.dat', help='path to predictor')
    parser.add_argument('--draw_landmarks', type=str2bool, default=False, help='choose to draw landmarks')
    parser.add_argument('--draw_video', type=str2bool, default=False, help='choose to draw video feed')
    parser.add_argument('--draw_detected_only', type=str2bool, default=False, help='choose to draw only when faces are detected')
    parser.add_argument('avatar', help='path to avatar folder')

    args = parser.parse_args()

    main(args.predictor, args.avatar, args.draw_landmarks, args.draw_video, args.draw_detected_only)
