# config.py
import os

# Thư mục lưu dataset và graph
# Cho phép ghi đè bằng biến môi trường OUTPUT_DIR để dùng trong Docker
OUTPUT_DIR = os.environ.get(
	"OUTPUT_DIR",
	r"D:\\MultiVSL\\MultiVSL\\dataset\\stgcn_dataset"
)

# Chuẩn hóa đường dẫn (nếu cần) và tạo thư mục
OUTPUT_DIR = os.path.normpath(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)
