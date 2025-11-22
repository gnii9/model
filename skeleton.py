# skeleton.py
import cv2
import numpy as np
import mediapipe as mp
import os
import json
import re
import unicodedata
from tqdm import tqdm
import tempfile

FRAMES_DIR = r"D:\MultiVSL\MultiVSL\dataset\images"
OUTPUT_DIR = r"D:\MultiVSL\MultiVSL\dataset\skeleton"

# Tạo output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def normalize(s):
    """Chuẩn hóa Unicode string"""
    return unicodedata.normalize("NFC", s)

def natural_sort(l):
    """Sắp xếp tự nhiên"""
    convert = lambda t: int(t) if t.isdigit() else t
    alphanum = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum)

def extract_keypoints(results):
    """Trích xuất 258 keypoints"""
    # Pose
    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] 
                         for lm in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33 * 4)

    # Left hand
    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z]
                       for lm in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21 * 3)

    # Right hand
    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z]
                       for lm in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21 * 3)

    return np.concatenate([pose, lh, rh]).tolist()

def safe_cv2_imread(image_path):
    """
    Đọc ảnh an toàn với đường dẫn Unicode
    Sử dụng temporary file copy nếu cần
    """
    try:
        # Thử đọc trực tiếp
        img = cv2.imread(image_path)
        if img is not None:
            return img
        
        # Nếu không được, copy file sang temp path không unicode
        temp_dir = tempfile.gettempdir()
        temp_filename = f"temp_{os.path.basename(image_path)}"
        temp_path = os.path.join(temp_dir, temp_filename)
        
        # Copy file gốc sang temp
        import shutil
        shutil.copy2(image_path, temp_path)
        
        # Đọc từ temp
        img = cv2.imread(temp_path)
        
        # Xóa temp file
        try:
            os.remove(temp_path)
        except:
            pass
            
        return img
        
    except Exception as e:
        print(f"Lỗi đọc ảnh {image_path}: {e}")
        return None

def extract_skeleton_all():
    """Hàm chính trích xuất skeleton với fix Unicode"""
    
    print("🦴 BẮT ĐẦU TRÍCH XUẤT SKELETON...")
    print(f"Input: {FRAMES_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    
    # Khởi tạo MediaPipe
    mp_holistic = mp.solutions.holistic.Holistic(
        static_image_mode=True,
        model_complexity=1,
        smooth_landmarks=True,
        refine_face_landmarks=False
    )
    
    total_classes = 0
    total_videos = 0
    total_frames = 0
    failed_videos = []
    
    # Lấy danh sách class folders
    class_folders = [f for f in os.listdir(FRAMES_DIR) 
                    if os.path.isdir(os.path.join(FRAMES_DIR, f))]
    
    print(f"Tổng số class: {len(class_folders)}")
    
    for label in class_folders:
        label_path = os.path.join(FRAMES_DIR, label)
        label_norm = normalize(label)
        out_label_dir = os.path.join(OUTPUT_DIR, label_norm)
        os.makedirs(out_label_dir, exist_ok=True)

        print(f"\n Class: {label_norm}")
        total_classes += 1
        
        # Lấy danh sách video folders
        video_folders = [f for f in os.listdir(label_path) 
                        if os.path.isdir(os.path.join(label_path, f))]
        
        print(f" Số video: {len(video_folders)}")
        
        for video_folder in video_folders:
            video_path = os.path.join(label_path, video_folder)
            video_norm = normalize(video_folder)
            out_video_dir = os.path.join(out_label_dir, video_norm)
            os.makedirs(out_video_dir, exist_ok=True)

            # Lấy danh sách frame files
            frame_files = [f for f in os.listdir(video_path) 
                          if f.lower().endswith(".jpg")]
            frame_files = natural_sort(frame_files)
            
            if not frame_files:
                print(f"  {video_norm}: Không có frames")
                failed_videos.append(f"{label_norm}/{video_norm} - No frames")
                continue
            
            print(f" {video_norm}: {len(frame_files)} frames")
            
            video_frame_count = 0
            video_failed_frames = 0
            
            # Xử lý từng frame
            for frame_file in tqdm(frame_files, desc="      Frames", leave=False):
                frame_path = os.path.join(video_path, frame_file)
                
                try:
                    # Đọc ảnh an toàn
                    img = safe_cv2_imread(frame_path)
                    if img is None:
                        video_failed_frames += 1
                        continue
                    
                    # Resize và chuyển đổi
                    img = cv2.resize(img, (640, 480))
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Process với MediaPipe
                    results = mp_holistic.process(rgb)
                    keypoints = extract_keypoints(results)
                    
                    # Lưu keypoints
                    out_json = os.path.join(out_video_dir, frame_file.replace(".jpg", ".json"))
                    with open(out_json, "w", encoding="utf-8") as f:
                        json.dump(keypoints, f, ensure_ascii=False)
                    
                    video_frame_count += 1
                    total_frames += 1
                    
                except Exception as e:
                    video_failed_frames += 1
                    continue
            
            # Thông báo kết quả video
            if video_frame_count > 0:
                success_rate = (video_frame_count / len(frame_files)) * 100
                print(f" {video_norm}: {video_frame_count}/{len(frame_files)} frames ({success_rate:.1f}%)")
                total_videos += 1
                
                if video_failed_frames > 0:
                    print(f" {video_failed_frames} frames thất bại")
            else:
                print(f" {video_norm}: THẤT BẠI")
                failed_videos.append(f"{label_norm}/{video_norm}")
    
    # Đóng MediaPipe
    mp_holistic.close()
    
    # Summary
    print(f"\n HOÀN THÀNH TRÍCH XUẤT SKELETON")
    print(f"  Số class: {total_classes}")
    print(f"  Số video thành công: {total_videos}")
    print(f"  Số frames: {total_frames}")
    
    if failed_videos:
        print(f"\n Video thất bại ({len(failed_videos)}):")
        for failed in failed_videos[:10]:  # Hiển thị 10 cái đầu
            print(f"   - {failed}")
        if len(failed_videos) > 10:
            print(f"   ... và {len(failed_videos) - 10} video khác")

def rename_unicode_folders():
    """
    Đổi tên folders có ký tự Unicode sang không dấu
    Chỉ chạy nếu cần thiết
    """
    FRAMES_DIR = r"D:\MultiVSL\MultiVSL\dataset\images"
    
    print(" Đang đổi tên folders Unicode...")
    
    # Map các ký tự Unicode sang không dấu
    unicode_map = {
        'á': 'a', 'à': 'a', 'ả': 'a', 'ã': 'a', 'ạ': 'a',
        'ă': 'a', 'ắ': 'a', 'ằ': 'a', 'ẳ': 'a', 'ẵ': 'a', 'ặ': 'a',
        'â': 'a', 'ấ': 'a', 'ầ': 'a', 'ẩ': 'a', 'ẫ': 'a', 'ậ': 'a',
        'đ': 'd',
        'é': 'e', 'è': 'e', 'ẻ': 'e', 'ẽ': 'e', 'ẹ': 'e',
        'ê': 'e', 'ế': 'e', 'ề': 'e', 'ể': 'e', 'ễ': 'e', 'ệ': 'e',
        'í': 'i', 'ì': 'i', 'ỉ': 'i', 'ĩ': 'i', 'ị': 'i',
        'ó': 'o', 'ò': 'o', 'ỏ': 'o', 'õ': 'o', 'ọ': 'o',
        'ô': 'o', 'ố': 'o', 'ồ': 'o', 'ổ': 'o', 'ỗ': 'o', 'ộ': 'o',
        'ơ': 'o', 'ớ': 'o', 'ờ': 'o', 'ở': 'o', 'ỡ': 'o', 'ợ': 'o',
        'ú': 'u', 'ù': 'u', 'ủ': 'u', 'ũ': 'u', 'ụ': 'u',
        'ư': 'u', 'ứ': 'u', 'ừ': 'u', 'ử': 'u', 'ữ': 'u', 'ự': 'u',
        'ý': 'y', 'ỳ': 'y', 'ỷ': 'y', 'ỹ': 'y', 'ỵ': 'y',
        ' ': '_'
    }
    
    def remove_accents(text):
        for char, replacement in unicode_map.items():
            text = text.replace(char, replacement)
            text = text.replace(char.upper(), replacement.upper())
        return text
    
    renamed_count = 0
    
    for root, dirs, files in os.walk(FRAMES_DIR):
        for name in list(dirs):
            old_path = os.path.join(root, name)
            new_name = remove_accents(name)
            
            if new_name != name:
                new_path = os.path.join(root, new_name)
                try:
                    os.rename(old_path, new_path)
                    print(f"✅ Đã đổi: {name} -> {new_name}")
                    renamed_count += 1
                except Exception as e:
                    print(f" Lỗi đổi {name}: {e}")
    
    print(f" Đã đổi {renamed_count} folders")

if __name__ == "__main__":
    print(">>> Script started")
    print(">>> Loading MediaPipe...")
    
    # Option 1: Đổi tên folders trước (nếu cần)
    # rename_unicode_folders()
    
    # Option 2: Chạy trích xuất với fix Unicode
    extract_skeleton_all()