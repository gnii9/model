# extract_image.py
import os
import subprocess
import unicodedata
import glob

VIDEOS_DIR = r"D:\MultiVSL\MultiVSL\dataset\videos"
OUTPUT_DIR = r"D:\MultiVSL\MultiVSL\dataset\images"

def normalize(s):
    """Chuẩn hóa chuỗi Unicode để tránh lỗi tiếng Việt."""
    return unicodedata.normalize("NFC", s)

def safe_mkdir(path):
    path = normalize(path)
    os.makedirs(path, exist_ok=True)

def extract_frames_ffmpeg(video_path, out_dir):
    """
    Dùng FFmpeg để xuất từng frame:
    Ví dụ: frame_000001.jpg, frame_000002.jpg
    """
    video_path = normalize(video_path)
    out_dir = normalize(out_dir)

    safe_mkdir(out_dir)

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-qscale:v", "2",        # chất lượng ảnh
        os.path.join(out_dir, "frame_%06d.jpg"),
        "-hide_banner",
        "-loglevel", "error"
    ]

    subprocess.run(cmd, shell=False)


def process_all_videos():

    print("🚀 BẮT ĐẦU CẮT FRAME BẰNG FFMPEG...")

    labels = glob.glob(os.path.join(VIDEOS_DIR, "*"))

    for label_folder in labels:

        if not os.path.isdir(label_folder):
            continue
        
        label_name = normalize(os.path.basename(label_folder))
        print(f"\n Label: {label_name}")

        video_files = glob.glob(os.path.join(label_folder, "*.*"))

        for video_file in video_files:

            ext = video_file.lower()
            if not ext.endswith((".mp4", ".mov", ".avi", ".mkv")):
                continue

            video_file = normalize(video_file)
            video_name = os.path.splitext(os.path.basename(video_file))[0]

            print(f" Đang xử lý: {video_name}")

            out_dir = os.path.join(OUTPUT_DIR, label_name, video_name)

            extract_frames_ffmpeg(video_file, out_dir)

    print("\n HOÀN TẤT CẮT FRAME BẰNG FFMPEG")


if __name__ == "__main__":
    process_all_videos()