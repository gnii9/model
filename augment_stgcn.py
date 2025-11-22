import numpy as np
import os
from tqdm import tqdm
from collections import Counter
import math

# ================== CONFIG ==================
OUTPUT_DIR = r"D:\MultiVSL\MultiVSL\dataset\stgcn_dataset"
TARGET_SAMPLES_PER_CLASS = 100  # Mục tiêu: Mỗi từ sẽ có 100 mẫu (bao gồm cả gốc)
TARGET_FRAMES = 64
NOISE_STD = 0.01
SCALE_RANGE = (0.95, 1.05)
TRANSLATE_RANGE = 0.05

# ================== AUGMENT FUNCTION ==================
def augment_keypoints(sequence, target_frames=64, noise_std=0.01, scale_range=(0.95,1.05), translate_range=0.05):
    """
    Augment 1 sequence keypoints VSL (T, V, 3) - Giữ nguyên hàm này
    """
    seq = sequence.copy()
    T, V, C = seq.shape
    
    # Temporal sampling
    if T != target_frames:
        indices = np.linspace(0, T - 1, target_frames).astype(int)
        seq = seq[indices]
    
    # Gaussian noise
    seq += np.random.normal(0, noise_std, seq.shape)
    
    # Scale x, y
    scale_x = np.random.uniform(scale_range[0], scale_range[1])
    scale_y = np.random.uniform(scale_range[0], scale_range[1])
    seq[..., 0] *= scale_x
    seq[..., 1] *= scale_y
    
    # Translate x, y
    trans_x = np.random.uniform(-translate_range, translate_range)
    trans_y = np.random.uniform(-translate_range, translate_range)
    seq[..., 0] += trans_x
    seq[..., 1] += trans_y
    
    # Center pelvis (joint 0)
    center = seq[:, 0:1, :]
    seq = seq - center
    
    return seq

# ================== BALANCED AUGMENT DATASET ==================
def augment_dataset_balanced(data_path, labels_path, target_count=TARGET_SAMPLES_PER_CLASS):
    """
    Tạo dataset cân bằng: Mỗi class sẽ được augment cho đến khi đạt đủ target_count
    """
    print("🔄 Đang load dữ liệu gốc...")
    data = np.load(data_path)   # (N, C, T, V, M)
    labels = np.load(labels_path)
    
    N, C, T, V, M = data.shape
    
    # 1. Phân nhóm dữ liệu theo Class (Label)
    class_data_map = {} # {label: [index1, index2, ...]}
    for idx, label in enumerate(labels):
        if label not in class_data_map:
            class_data_map[label] = []
        class_data_map[label].append(idx)
        
    augmented_data = []
    augmented_labels = []
    
    print(f"Mục tiêu: {target_count} mẫu/class")
    print(f"Thống kê trước khi Augment: {dict(Counter(labels))}")

    # 2. Duyệt qua từng Class để xử lý
    sorted_labels = sorted(class_data_map.keys())
    
    for label in tqdm(sorted_labels, desc="Balancing Classes"):
        indices = class_data_map[label]
        current_count = len(indices)
        
        # Lấy dữ liệu gốc của class này
        class_samples = []
        for idx in indices:
            # Chuyển về (T, V, C) để dễ augment
            seq = data[idx].squeeze(-1).transpose(1, 2, 0)
            class_samples.append(seq)
            
            # Thêm luôn bản gốc vào danh sách kết quả
            augmented_data.append(data[idx])
            augmented_labels.append(label)
            
        # Tính toán số lượng cần sinh thêm
        needed = target_count - current_count
        
        if needed <= 0:
            # Nếu class này đã có đủ hoặc thừa mẫu thì thôi (hoặc có thể cắt bớt nếu muốn)
            continue
            
        # Sinh thêm dữ liệu
        # Để sinh đủ 'needed' mẫu từ 'current_count' video gốc, ta chia đều
        generated_count = 0
        i = 0
        
        while generated_count < needed:
            # Lấy video gốc theo vòng tròn (Round-robin) để augment đều các mẫu gốc
            original_seq = class_samples[i % current_count]
            
            aug_seq = augment_keypoints(original_seq, target_frames=T, 
                                        noise_std=NOISE_STD,
                                        scale_range=SCALE_RANGE,
                                        translate_range=TRANSLATE_RANGE)
            
            # Chuyển về lại format ST-GCN (C, T, V, M)
            aug_seq_stgcn = aug_seq.transpose(2, 0, 1)[..., np.newaxis]
            
            augmented_data.append(aug_seq_stgcn)
            augmented_labels.append(label)
            
            generated_count += 1
            i += 1

    # Convert thành np.array
    augmented_data = np.array(augmented_data)
    augmented_labels = np.array(augmented_labels)
    
    # Lưu ra file
    np.save(os.path.join(OUTPUT_DIR, 'stgcn_data_aug.npy'), augmented_data)
    np.save(os.path.join(OUTPUT_DIR, 'stgcn_labels_aug.npy'), augmented_labels)
    
    print(f"\n✅ Dataset augmented saved successfully!")
    print(f"Original samples: {N}")
    print(f"Total samples after balancing: {len(augmented_data)}")
    
    # Kiểm tra lại độ cân bằng
    print(f"📊 Thống kê sau khi Augment (Sample): {dict(list(Counter(augmented_labels).items())[:5])} ...")

# ================== RUN ==================
if __name__ == "__main__":
    data_path = os.path.join(OUTPUT_DIR, 'stgcn_data.npy')
    labels_path = os.path.join(OUTPUT_DIR, 'stgcn_labels.npy')
    
    # Chạy hàm cân bằng thay vì hàm cũ
    augment_dataset_balanced(data_path, labels_path, target_count=100)