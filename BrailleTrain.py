import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

# Define dataset
class BrailleDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.images = []
        self.labels = []
        self.transform = transform
        self.alp = list("abcdefghijklmnopqrstuvwxyz")
        
        for dirName, _, fileList in os.walk(root_dir):
            for fname in fileList:
                path = os.path.join(dirName, fname)
                self.images.append(path)
                # Label is first char converted to index 0-based
                label_char = fname[0].lower()
                if label_char in self.alp:
                    self.labels.append(self.alp.index(label_char))
                else:
                    self.labels.append(-1)  # invalid
                
        # Filter out invalids
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

# Transformations including resize and normalization
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),  # Converts to [0,1] and CHW tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load full dataset
dataset = BrailleDataset("Braille_Dataset", transform=transform)

# Split dataset: 70% train, 15% val, 15% test
total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define CNN Model
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
model = BrailleCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Training loop with validation
epochs = 30

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_acc = 100 * val_correct / val_total

    print(f"Epoch {epoch+1}/{epochs} "
          f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% "
          f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%")

# Final evaluation on test set after training completes
model.eval()
test_loss = 0.0
test_correct = 0
test_total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_loss /= len(test_loader)
test_acc = 100 * test_correct / test_total

print(f"\nFinal Test Loss: {test_loss:.4f} Final Test Accuracy: {test_acc:.2f}%")

# Save the trained model
torch.save(model.state_dict(), "Braille.pth")
print("Model saved as Braille.pth")