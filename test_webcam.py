import cv2
import numpy as np
import torch
import mediapipe as mp
import json
from st_gcn_model import STGCN

# ======================
# CONFIG
# ======================
MODEL_PATH = r"D:\MultiVSL\MultiVSL\dataset\stgcn_dataset\stgcn_mediapipe75.pth"
LABEL_MAP_PATH = r"D:\MultiVSL\MultiVSL\dataset\stgcn_dataset\label_map.json"
GRAPH_PATH = r"D:\MultiVSL\MultiVSL\dataset\stgcn_dataset\mediapipe_graph.npy"

TARGET_FRAMES = 64
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load label map
with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
    label_map = json.load(f)
inv_label_map = {v: k for k, v in label_map.items()}

# Load model
A = np.load(GRAPH_PATH)
model = STGCN(in_channels=3, num_class=len(label_map), A=A)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

print("Model loaded, using device:", device)

# MediaPipe setup
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_keypoints(results):
    # Lấy pose
    pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten() \
           if results.pose_landmarks else np.zeros(33*3)
    lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten() \
         if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten() \
         if results.right_hand_landmarks else np.zeros(21*3)

    kps = np.concatenate([pose, lh, rh])  # (225,)

    # --- Normalization ---
    kps = kps.reshape(-1,3)  # (75,3)

    # Center by hip / pelvis (pose landmark 0)
    center = kps[0].copy()
    kps -= center

    # Scale: normalize by torso length (shoulder distance)
    shoulder_dist = np.linalg.norm(kps[11] - kps[12])  # pose landmarks L/R shoulders
    if shoulder_dist > 0:
        kps /= shoulder_dist

    return kps.flatten()


seq_buffer = []
pred_label = ""
conf = 0.0

# ======================
# START WEBCAM
# ======================
cap = cv2.VideoCapture(0)

print("🎥 Webcam ready... Press 'q' to quit.")
seq_np = np.zeros((TARGET_FRAMES, 75, 3))  # placeholder

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không đọc được frame từ camera")
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(img_rgb)
    mp_drawing = mp.solutions.drawing_utils
    mp_style = mp.solutions.drawing_styles

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_style.get_default_pose_landmarks_style())
    
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS)

    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS)


    keypoints = extract_keypoints(results)
    seq_buffer.append(keypoints)

    if len(seq_buffer) > TARGET_FRAMES:
        seq_buffer = seq_buffer[-TARGET_FRAMES:]

    if len(seq_buffer) == TARGET_FRAMES:
        seq_np = np.array(seq_buffer)

        try:
            seq_np = seq_np.reshape(TARGET_FRAMES, 75, 3)
        except:
            print("ERROR reshape — keypoints lỗi / thiếu!")
            continue

        seq_np = seq_np.transpose(2, 0, 1)[..., np.newaxis]
        x = torch.FloatTensor(seq_np).unsqueeze(0).to(device) 

        with torch.no_grad():
            out = model(x)
            pred = out.argmax(dim=1).item()
            pred_label = inv_label_map[pred]
            conf = torch.softmax(out, dim=1)[0][pred].item()

    # ===== DEBUG =====


    # ======= DRAW TEXT =======
    # print("keypoints min/max:", seq_np.min(), seq_np.max())

    cv2.putText(frame, f"Conf: {conf:.2f}", (20,150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)

    cv2.putText(frame, f"Frames: {len(seq_buffer)}", (20,50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

    cv2.putText(frame, f"Predict: {pred_label}", (20,100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)

    cv2.imshow("VSL - STGCN Real-Time", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
