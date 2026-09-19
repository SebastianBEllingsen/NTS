import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time


model_path='/home/sebastian/projects/NTS/models/hand_landmarker.task'
BaseOptions = mp.tasks.BaseOptions


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

def main():
    while True:
        attempt = 0
        success, frame = cap.read()
        #print(success)
        while not success and attempt < 5:
            time.sleep(0.2)
            success, frame = cap.read()
            attempt += 1
        if not success:
            print("Failed to read frame")
            break

        img = cv2.flip(frame, 1)
        h, w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()