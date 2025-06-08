import cv2
import dlib
import torch
import numpy as np
from train_model import GazeCNN
from feature_extraction import extract_eye
from utils import draw_gaze_direction, calculate_gaze_vector

PREDICTOR_PATH = "../models/shape_predictor_68_face_landmarks.dat"
MODEL_PATH = "../models/gaze_regressor.pth"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GazeCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

LEFT_EYE_POINTS = list(range(36, 42))
RIGHT_EYE_POINTS = list(range(42, 48))

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)

            left_eye_img = extract_eye(frame, landmarks, LEFT_EYE_POINTS)
            right_eye_img = extract_eye(frame, landmarks, RIGHT_EYE_POINTS)

            # Prepare tensors
            left_eye_tensor = torch.tensor(left_eye_img).unsqueeze(0).unsqueeze(0).to(device)
            right_eye_tensor = torch.tensor(right_eye_img).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                gaze_left = model(left_eye_tensor).cpu().numpy()[0]
                gaze_right = model(right_eye_tensor).cpu().numpy()[0]
                gaze = (gaze_left + gaze_right) / 2

            yaw, pitch = gaze
            cv2.putText(frame, f"Yaw: {yaw:.2f}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Pitch: {pitch:.2f}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Draw eye landmarks
            for i in range(36, 48):
                pt = landmarks.part(i)
                cv2.circle(frame, (pt.x, pt.y), 2, (255, 0, 0), -1)

            # Draw gaze direction arrow using utils
            gaze_vec = calculate_gaze_vector(yaw, pitch)
            frame = draw_gaze_direction(frame, gaze_vec, origin=(frame.shape[1]//2, frame.shape[0]//2))

        cv2.imshow("Real-time Gaze Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

