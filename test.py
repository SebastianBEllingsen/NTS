import time
import cv2
import mediapipe as mp

# Short aliases for the Modern Tasks API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Global variable to store tracking results from the background thread
latest_result = None

# MediaPipe Hand Connections mapping (defines which joints connect to each other)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # Index finger
    (9, 10), (10, 11), (11, 12),         # Middle finger
    (13, 14), (14, 15), (15, 16),        # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (5, 9), (9, 13), (13, 17)            # Palm base connection
]

def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

def main():
    global latest_result
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Configuration for the Live Stream
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='/home/sebastian/projects/NTS/models/hand_landmarker.task'),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=print_result,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Initialize the detector
    with HandLandmarker.create_from_options(options) as landmarker:
        print("Running... Press 'q' to quit.")
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Asynchronous detection requires a millisecond timestamp
            frame_timestamp_ms = int(time.time() * 1000)
            landmarker.detect_async(mp_image, frame_timestamp_ms)

            # Draw landmarks using pure OpenCV
            if latest_result is not None and latest_result.hand_landmarks:
                for hand_landmarks in latest_result.hand_landmarks:
                    # 1. Convert normalized coordinates to absolute pixel coordinates
                    pixel_coords = []
                    for lm in hand_landmarks:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        pixel_coords.append((cx, cy))
                    
                    # 2. Draw connection lines
                    for connection in HAND_CONNECTIONS:
                        start_idx, end_idx = connection
                        if start_idx < len(pixel_coords) and end_idx < len(pixel_coords):
                            cv2.line(frame, pixel_coords[start_idx], pixel_coords[end_idx], (0, 255, 0), 2)
                    
                    # 3. Draw joint dots
                    for coord in pixel_coords:
                        cv2.circle(frame, coord, 5, (0, 0, 255), cv2.FILLED)

            cv2.imshow('MediaPipe Hands', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
