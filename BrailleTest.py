import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import cv2
import os

# Re-define your BrailleDataset to load test data
class BrailleDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.images = []
        self.labels = []
        self.transform = transform
        self.alp = list("abcdefghijklmnopqrstuvwxyz")

        for dirName, _, fileList in os.walk(root_dir):
            for fname in fileList:
                path = os.path.join(dirName, fname)
                self.images.append(path)
                label_char = fname[0].lower()
                if label_char in self.alp:
                    self.labels.append(self.alp.index(label_char))
                else:
                    self.labels.append(-1)

        filtered = [(img, lbl) for img, lbl in zip(self.images, self.labels) if lbl >= 0]
        self.images, self.labels = zip(*filtered)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

# CNN Model definition (same as training)
class BrailleCNN(nn.Module):
    def __init__(self):
        super(BrailleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 28, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 13 * 13, 128)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 26)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = BrailleCNN().to(device)
model.load_state_dict(torch.load("Braille.pth", map_location=device))
model.eval()

# Define the same transforms used during training
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load full dataset again
dataset = BrailleDataset("Braille_Dataset", transform=transform)

# Split same way as training (you can hardcode sizes or save splits)
total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

_, _, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

alp = list("abcdefghijklmnopqrstuvwxyz")

# Evaluate on test dataset
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1)
        predicted = torch.argmax(probs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy on split test dataset: {100 * correct / total:.2f}%")