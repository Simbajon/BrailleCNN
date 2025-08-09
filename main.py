# main.py

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os

ALPHABET = list("abcdefghijklmnopqrstuvwxyz")
IMG_SIZE = (28, 28)

class BrailleCNN(nn.Module):
    def __init__(self):
        super(BrailleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 28->14
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 14->7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 26)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def preprocess_image(image_path, device):
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    return img_tensor

def is_image_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading Braille Recognition model...")
    model = BrailleCNN().to(device)
    try:
        model.load_state_dict(torch.load("best_braille_cnn.pth", map_location=device))
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return
    model.eval()
    print("Model loaded successfully.")

    while True:
        print("\nOptions:")
        print("1. Enter path to folder of Braille images")
        print("2. Exit")
        choice = input("Select option (1/2): ").strip()

        if choice == '1':
            folder_path = input("Enter folder path: ").strip()
            if not os.path.isdir(folder_path):
                print("Invalid folder path. Please try again.")
                continue

            image_files = [f for f in os.listdir(folder_path) if is_image_file(f)]
            if not image_files:
                print("No image files found in the folder.")
                continue

            for img_file in image_files:
                img_path = os.path.join(folder_path, img_file)
                try:
                    img_tensor = preprocess_image(img_path, device)
                except Exception as e:
                    print(f"Error loading image {img_file}: {e}")
                    continue

                with torch.no_grad():
                    outputs = model(img_tensor)
                    pred_idx = torch.argmax(outputs, dim=1).item()

                print(f"{img_file} -> Predicted Braille character: {ALPHABET[pred_idx]}")

                img_cv = cv2.imread(img_path)
                if img_cv is not None:
                    cv2.imshow(f"Image: {img_file}", img_cv)
                    print("Press any key on the image window to continue...")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                else:
                    print(f"Could not open image {img_file} for display.")

        elif choice == '2':
            print("Exiting program. Goodbye!")
            break
        else:
            print("Invalid option. Please enter 1 or 2.")

if __name__ == "__main__":
    main()