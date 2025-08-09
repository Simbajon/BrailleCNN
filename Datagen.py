# Datagen.py

import os
from PIL import Image
import torchvision.transforms as transforms

rootDir = "Brail"
save_dir = "Braille_Dataset"
os.makedirs(save_dir, exist_ok=True)

# Define augmentations as torchvision transforms
brightness_transform = transforms.ColorJitter(brightness=(0.2, 1.0))
shift_transform = transforms.RandomAffine(degrees=0, translate=(0.2, 0.2))
rotation_transform = transforms.RandomRotation(degrees=90)

resize = transforms.Resize((28, 28))

# Allowed image extensions (lowercase)
img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

for dirName, subdirList, fileList in os.walk(rootDir):
    for fname in fileList:
        ext = os.path.splitext(fname)[1].lower()
        if ext not in img_extensions:
            continue  # skip non-image files

        img_path = os.path.join(dirName, fname)
        img = Image.open(img_path).convert('RGB')
        img = resize(img)

        filename_no_ext = os.path.splitext(fname)[0]

        for i in range(20):
            # Brightness variation
            bright_img = brightness_transform(img)
            bright_img.save(f"{save_dir}/{filename_no_ext}_{i}_dim.jpg")

            # Width/height shift
            shifted_img = shift_transform(img)
            shifted_img.save(f"{save_dir}/{filename_no_ext}_{i}_whs.jpg")

            # Rotation
            rotated_img = rotation_transform(img)
            rotated_img.save(f"{save_dir}/{filename_no_ext}_{i}_rot.jpg")