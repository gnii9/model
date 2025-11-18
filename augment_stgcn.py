import numpy as np
import os
from tqdm import tqdm

# ================== CONFIG ==================
OUTPUT_DIR = r"D:\MultiVSL\MultiVSL\dataset\stgcn_dataset"
AUG_FACTOR = 10          # số lần augment mỗi video
TARGET_FRAMES = 64
NOISE_STD = 0.01
SCALE_RANGE = (0.95, 1.05)
TRANSLATE_RANGE = 0.05

# ================== AUGMENT FUNCTION ==================
def augment_keypoints(sequence, target_frames=64, noise_std=0.01, scale_range=(0.95,1.05), translate_range=0.05):
    """
    Augment 1 sequence keypoints VSL (T, V, 3)
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

# ================== AUGMENT DATASET ==================
def augment_dataset(data_path, labels_path, aug_factor=AUG_FACTOR):
    """
    Tạo dataset augmented từ stgcn_data.npy và labels
    """
    data = np.load(data_path)  # (N, C, T, V, M)
    labels = np.load(labels_path)
    
    N, C, T, V, M = data.shape
    augmented_data = []
    augmented_labels = []
    
    for i in tqdm(range(N), desc="Augmenting dataset"):
        # Chuyển về (T, V, C)
        seq = data[i].squeeze(-1).transpose(1, 2, 0)  # (T, V, 3)
        
        # Thêm bản gốc
        augmented_data.append(data[i])
        augmented_labels.append(labels[i])
        
        # Augment N lần
        for _ in range(aug_factor):
            aug_seq = augment_keypoints(seq, target_frames=T, 
                                        noise_std=NOISE_STD,
                                        scale_range=SCALE_RANGE,
                                        translate_range=TRANSLATE_RANGE)
            # Chuyển về (C, T, V, M)
            aug_seq_stgcn = aug_seq.transpose(2, 0, 1)[..., np.newaxis]
            augmented_data.append(aug_seq_stgcn)
            augmented_labels.append(labels[i])
    
    # Convert thành np.array
    augmented_data = np.array(augmented_data)
    augmented_labels = np.array(augmented_labels)
    
    # Lưu ra
    np.save(os.path.join(OUTPUT_DIR, 'stgcn_data_aug.npy'), augmented_data)
    np.save(os.path.join(OUTPUT_DIR, 'stgcn_labels_aug.npy'), augmented_labels)
    
    print(f"\nDataset augmented saved")
    print(f"Original: {N} samples")
    print(f"Augmented: {len(augmented_data)} samples")
    
    return augmented_data, augmented_labels

# ================== RUN ==================
if __name__ == "__main__":
    data_path = os.path.join(OUTPUT_DIR, 'stgcn_data.npy')
    labels_path = os.path.join(OUTPUT_DIR, 'stgcn_labels.npy')
    augment_dataset(data_path, labels_path)
