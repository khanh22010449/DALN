import os
import json
import cv2
import matplotlib.pyplot as plt


def create_annotation(path, output_dir, size=(256, 256)):
    """Load data from path and resize images"""
    dataset = []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for folder_name in os.listdir(path):
        folder_path = os.path.join(path, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                data = {}

                file_path = os.path.join(folder_path, file_name)
                if os.path.isfile(file_path):
                    img = cv2.imread(file_path)
                    if img is not None:
                        # Resize the image
                        resized_img = cv2.resize(img, size)

                        # Save the resized image
                        label_dir = os.path.join(output_dir, folder_name)
                        if not os.path.exists(label_dir):
                            os.makedirs(label_dir)
                        output_path = os.path.join(label_dir, file_name)
                        cv2.imwrite(output_path, resized_img)
                        print(f"Saved resized image to {output_path}")

                        # Add metadata to the dataset
                        data["path_file"] = output_path
                        data["label"] = folder_name
                        dataset.append(data)

    # Save dataset annotations to a JSON file
    with open("./DALN/Annotations.json", "w") as f:
        for item in dataset:
            sample = {"path_file": item["path_file"], "label": item["label"]}
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def show_img(path):
    """Show image from path"""
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def check_resize(path):
    with open(path, "r") as f:
        for line in f:
            data = json.loads(line)
            img_path = data["path_file"]
            img = cv2.imread(img_path)
            print(f"size image: {img.shape}")


# Load data, resize images, and save resized images
create_annotation("./Dataset/train", "./Resized_Images", size=(128, 128))

# Check the size of the resized images
check_resize("./DALN/Annotations.json")
