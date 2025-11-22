# Báo Cáo Dự Án: MultiVSL (tóm tắt code)

## 1. Mục đích dự án

Dự án này xây dựng một pipeline nhận diện/nhận dạng video/gesture cho Ngôn ngữ Ký hiệu (VSL) dựa trên biểu diễn keypoints (skeleton) trích xuất từ MediaPipe, tiền xử lý và huấn luyện mô hình ST-GCN (Spatio-Temporal Graph Convolutional Network). Mục tiêu: chuyển video thành chuỗi keypoints, augment dữ liệu, huấn luyện ST-GCN và dùng model chạy realtime trên webcam.

## 2. Kỹ thuật & thư viện chính

- **Ngôn ngữ:** Python
- **Deep learning:** PyTorch (torch, torch.nn)
- **Keypoint extraction:** MediaPipe (holistic)
- **Xử lý ảnh và IO:** OpenCV, FFmpeg (qua subprocess), numpy, json
- **Tiện ích:** tqdm, pathlib, tempfile
- **Mô hình:** ST-GCN (GraphConv + TemporalConv + STGCN_Block)

## 3. Các file chính và vai trò

- `extract_image.py`: Dùng FFmpeg để tách frame ra từ các video trong `dataset/videos` sang `dataset/images`. Có xử lý chuẩn hóa Unicode để tránh lỗi tên file/đường dẫn tiếng Việt.
- `skeleton.py`: Dùng MediaPipe Holistic để trích xuất keypoints (pose, left hand, right hand) từ mỗi frame, lưu thành file `.json` theo cấu trúc folder tương ứng. Chứa các tiện ích an toàn với đường dẫn Unicode và hàm `rename_unicode_folders()` để đổi tên nếu cần.
- `augment_stgcn.py`: Chứa logic augment dữ liệu trên biểu diễn keypoints (temporal sampling, gaussian noise, scaling, translation), và hàm `augment_dataset_balanced` để cân bằng số mẫu theo class, lưu ra `stgcn_data_aug.npy` và `stgcn_labels_aug.npy`.
- `st_gcn_model.py`: Định nghĩa kiến trúc ST-GCN bao gồm:
  - `GraphConv`: phép chuyển không gian (sử dụng ma trận adjacency `A`)
  - `TemporalConv`: convolution dọc theo trục thời gian (kernel size mặc định 9)
  - `STGCN_Block`: kết hợp spatial + temporal + residual
  - `STGCN`: stack các block, pooling và fully-connected cuối để phân loại
- `train_stgcn.py`: Dataset wrapper, training loop, validation và early stopping. Tải `mediapipe_graph.npy`, `label_map.json`, dùng dữ liệu `stgcn_data_aug.npy` để huấn luyện, lưu model tốt nhất và model cuối cùng vào thư mục `OUTPUT_DIR`.
- `test_webcam.py`: Pipeline realtime: lấy keypoints từ MediaPipe, gom thành chuỗi `TARGET_FRAMES`, tiền xử lý (center, scale), đưa vào model `STGCN` đã load từ file `.pth`, hiển thị dự đoán và độ tin cậy lên khung hình webcam.
- `config.py`: Định nghĩa `OUTPUT_DIR` và tạo thư mục nếu chưa tồn tại.

## 4. Dữ liệu & file mong đợi

- `dataset/images/`: frames tách từ video theo folder nhãn / video
- `dataset/skeleton/`: json keypoints cho từng frame
- `dataset/stgcn_dataset/stgcn_data.npy` (hoặc `stgcn_data_aug.npy`): dữ liệu dạng `(N, C, T, V, M)` hoặc tương tự dùng cho model
- `dataset/stgcn_dataset/mediapipe_graph.npy`: ma trận adjacency / graph
- `dataset/stgcn_dataset/label_map.json`: map label <-> index
- models: `stgcn_mediapipe75.pth`, `stgcn_best.pth`

## 5. Cách chạy (tổng quan)

1. Tách frames từ video:
   - Chỉnh `VIDEOS_DIR` và `OUTPUT_DIR` trong `extract_image.py` nếu cần.
   - Chạy: `python extract_image.py` (yêu cầu ffmpeg cài sẵn và trong PATH).

2. Trích xuất skeleton (keypoints) từ frames:
   - Chạy: `python skeleton.py` (cần cài `mediapipe`, `opencv-python`)

3. Augment & chuẩn bị dataset ST-GCN:
   - Chạy: `python augment_stgcn.py` (sẽ đọc `stgcn_data.npy`/`stgcn_labels.npy` và lưu `stgcn_data_aug.npy`)

4. Huấn luyện model ST-GCN:
   - Chạy: `python train_stgcn.py` (cần PyTorch, GPU nếu có để nhanh hơn)

5. Kiểm thử realtime:
   - Chạy: `python test_webcam.py` để dùng webcam realtime với model đã huấn luyện.

## 6. Ghi chú kỹ thuật & phát hiện vấn đề

- `st_gcn_model.py`: có comment chỉ ra một số bước chuyển đổi kích thước (permute/view) để phù hợp BatchNorm; cấu trúc chung hiển thị hợp lý.
- Dữ liệu input cho model trong `test_webcam.py` được dựng thành `(1, 3, T, V, 1)` trước khi đẩy vào model — hãy đảm bảo conformance với shape mong đợi `(N, C, T, V, M)`.
- `skeleton.py` lưu keypoints dạng list flatten (pose + lh + rh). Khi dựng dataset cho ST-GCN cần ánh xạ/reshape cho đúng `(C, T, V, M)`.
- Đường dẫn (path) hiện đang hard-coded tuyệt đối (ví dụ `D:\MultiVSL\MultiVSL\dataset\...`). Khuyến nghị di chuyển vào `config.py` hoặc dùng biến môi trường để dễ chia sẻ và deploy.

## 7. Gợi ý cải tiến (prioritized)

1. Tạo `requirements.txt` hoặc `pyproject.toml` liệt kê các package: `torch`, `opencv-python`, `mediapipe`, `numpy`, `tqdm` để tiện cài đặt môi trường.
2. Rút cấu hình đường dẫn ra `config.py` hoặc dùng `argparse` để override khi chạy script.
3. Thêm kiểm tra shape rõ ràng và raising error sớm khi dữ liệu không đúng shape.
4. Thêm script/command để chuyển đổi từ JSON keypoints (skeleton) sang npy dataset cho ST-GCN, kèm mapping joints → V index rõ ràng.
5. Thêm logging (thay vì prints) và unit tests cơ bản cho các hàm tiền xử lý (augment, extract_keypoints, reshape conversions).

## 8. Kết luận ngắn

Codebase đã có full pipeline từ video → frames → keypoints → dataset → augment → train → realtime inference. Cần một số bước cấu hình (paths, requirements) và kiểm tra shape/format dữ liệu để reproducible và dễ dùng trên máy khác.

---

Nếu muốn, tôi có thể:
- Sinh `requirements.txt` tự động từ code,
- Viết script chuyển JSON keypoints → `.npy` dataset phù hợp ST-GCN,
- Hoặc tách đường dẫn ra `config.py`/`args` để dễ cấu hình.

Report này được tạo tự động dựa trên nội dung file hiện có trong repo.
