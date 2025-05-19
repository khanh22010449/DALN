import os
import json
import matplotlib.pyplot as plt
from PIL import Image
import logging

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose, Resize


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_data(json_path: str):
    """Load dataset từ file JSON với transform và tạo DataLoader."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"File JSON không tồn tại: {json_path}")

    def data_generator():
        with open(json_path, "r") as f:
            for line in f:
                data = json.loads(line)
                img_path = data.get("path_file")

                if not os.path.exists(img_path):
                    logging.warning(f"Đường dẫn ảnh không tồn tại: {img_path}")
                    continue

                try:
                    img = Image.open(img_path).convert("RGB")  # Load ảnh bằng PIL
                    yield {"image": img, "label": data.get("label")}

                except Exception as e:
                    logging.error(f"Lỗi khi load ảnh từ {img_path}: {e}")
                    continue  # Bỏ qua ảnh lỗi, tiếp tục ảnh khác

    dataset = list(data_generator())

    if not dataset:
        raise ValueError(
            "Không tìm thấy ảnh hợp lệ. Hãy kiểm tra đường dẫn và file JSON."
        )

    return dataset


def transform_image(dataset):
    """Chuyển đổi ảnh trong dataset thành tensor và chuẩn hóa."""
    transform = Compose(
        [
            Resize((128, 128)),  # Resize ảnh về kích thước chuẩn
            ToTensor(),  # Chuyển ảnh sang tensor
            Normalize(  # Chuẩn hóa giống ImageNet
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    for data in dataset:
        data["image"] = transform(data["image"])

    return dataset


if __name__ == "__main__":
    # Thay đường dẫn bằng file JSON thật của bạn
    json_path = "./DALN/Annotations.json"

    dataset = load_data(json_path)
    train_loadder = transform_image(dataset)
