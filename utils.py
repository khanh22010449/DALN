import os
import json
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose, Resize


def load_data(json_path):
    """Load dataset từ file JSON với transform và tạo DataLoader."""
    dataset = []

    # Transform cho ảnh: resize, tensor, normalize
    transform = Compose(
        [
            Resize((256, 256)),  # Resize ảnh về kích thước chuẩn
            ToTensor(),  # Chuyển ảnh sang tensor
            Normalize(  # Chuẩn hóa giống ImageNet
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    with open(json_path, "r") as f:
        for line in f:
            data = json.loads(line)
            img_path = data["path_file"]

            try:
                img = Image.open(img_path).convert("RGB")  # Load ảnh bằng PIL

                img_tensor = transform(img)  # Áp dụng transform

                # Thêm ảnh và nhãn vào dataset
                dataset.append({"image": img_tensor, "label": data["label"]})

            except Exception as e:
                print(f"Lỗi khi load ảnh từ {img_path}: {e}")
                continue  # Bỏ qua ảnh lỗi, tiếp tục ảnh khác

    if len(dataset) == 0:
        raise ValueError(
            "Không tìm thấy ảnh hợp lệ. Hãy kiểm tra đường dẫn và file JSON."
        )

    # Tạo DataLoader
    train_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,  # Để 0 nếu đang debug hoặc dùng Windows
    )

    # Debug: In thông tin 1 batch
    for batch in train_loader:
        images = batch["image"]
        labels = batch["label"]
        print(
            f"✅ Batch size: {images.size(0)}, Image shape: {images.shape}, Labels: {labels}"
        )
        break  # Chỉ in batch đầu tiên để kiểm tra

    return train_loader


def show_image(tensor_img, label=None):
    """Hiển thị ảnh từ tensor (sau transform)"""
    img = tensor_img.clone()
    img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(3, 1, 1)
    img = img.clamp(0, 1)
    img = img.permute(1, 2, 0)  # CHW -> HWC
    plt.imshow(img)
    if label is not None:
        plt.title(f"Label: {label}")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    # Thay đường dẫn bằng file JSON thật của bạn
    json_path = "./DALN/Annotations.json"

    train_loader = load_data(json_path)

    # Hiển thị ảnh đầu tiên trong dataset
    first_item = train_loader[0]
    print("Loại dữ liệu:", type(first_item))
    show_image(first_item["image"], label=first_item["label"])
