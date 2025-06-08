# Helper functions
import numpy as np
import cv2

def smooth_gaze(gaze_angles, alpha=0.7):
    smoothed = np.zeros_like(gaze_angles)
    smoothed[0] = gaze_angles[0]
    for i in range(1, len(gaze_angles)):
        smoothed[i] = alpha * gaze_angles[i] + (1 - alpha) * smoothed[i-1]
    return smoothed

def draw_gaze_direction(frame, gaze_vector, origin=None, length=100, color=(0, 255, 0), thickness=2):
    h, w = frame.shape[:2]
    if origin is None:
        origin = (w // 2, h // 2)
    end_point = (
        int(origin[0] + length * gaze_vector[0]),
        int(origin[1] - length * gaze_vector[1])  # y axis inverted in images
    )
    cv2.arrowedLine(frame, origin, end_point, color, thickness, tipLength=0.2)
    return frame

def calculate_gaze_vector(yaw, pitch):
    x = -np.sin(yaw)
    y = -np.sin(pitch)
    norm = np.sqrt(x ** 2 + y ** 2)
    if norm == 0:
        return (0, 0)
    return (x / norm, y / norm)
