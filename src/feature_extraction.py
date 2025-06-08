import cv2
import numpy as np
import dlib

predictor_path = "../models/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

def preprocess_eye_image(img, target_size=(60, 36)):
    """
    Preprocess eye image: grayscale, resize, normalize to [0,1].
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img, target_size)
    img_normalized = img_resized.astype(np.float32) / 255.0
    return img_normalized

def extract_eye(image, landmarks, eye_points):
    pts = [landmarks.part(i) for i in eye_points]
    x_min = min(p.x for p in pts)
    y_min = min(p.y for p in pts)
    x_max = max(p.x for p in pts)
    y_max = max(p.y for p in pts)
    eye_img = image[y_min:y_max, x_min:x_max]
    eye_img = preprocess_eye_image(eye_img)
    return eye_img
