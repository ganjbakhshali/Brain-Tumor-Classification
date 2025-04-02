import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import matplotlib.pyplot as plt

def plot_history(train_losses, val_losses, train_accs, val_accs):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.show()


class AreaAttentionLayer(nn.Module):
    def __init__(self, in_channels, reduction_ratio: int = 8):
        super(AreaAttentionLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // reduction_ratio, 1, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(in_channels // reduction_ratio)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        attention = self.conv1(x)
        attention = self.bn(attention)
        attention = F.relu(attention)
        attention = self.conv2(attention)
        attention = torch.sigmoid(attention)

        return x * attention

class CNNBlock(nn.Module):
    def __init__(self, filters: int, use_attention: bool = True):
        super(CNNBlock, self).__init__()
        self.use_attention = use_attention
        
        if self.use_attention:
            self.area_attention = AreaAttentionLayer(filters)
        
        self.conv = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters)
        )
        
        self.shortcut = nn.Identity()

    def forward(self, x):
        identity = x
        
        if self.use_attention:
            x = self.area_attention(x)
        
        x = self.conv(x)
        
        if x.shape[1] != identity.shape[1]:
            identity = nn.Conv2d(identity.shape[1], x.shape[1], kernel_size=1, bias=False)(identity)
            identity = nn.BatchNorm2d(x.shape[1])(identity)
        
        x += identity
        return F.relu(x)

class RESNet(nn.Module):
    def __init__(self, num_classes: int = 1, dropout_rate: float = 0.3, initial_filters: int = 32):  # num_classes=1
        super(RESNet, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, initial_filters, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(initial_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.blocks = nn.ModuleList([CNNBlock(initial_filters * (2**i)) for i in range(3)])
        
        self.downsamples = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(initial_filters * (2**i), initial_filters * (2**(i+1)), kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(initial_filters * (2**(i+1))),
                nn.ReLU(inplace=True)
            ) for i in range(2)
        ])
        
        self.final_attention = AreaAttentionLayer(initial_filters * 4)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(initial_filters * 4, num_classes)  # num_classes=1
    
    def forward(self, x):
        x = self.stem(x)
        
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)
                x = self.dropout(x)
        
        x = self.final_attention(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x #.squeeze(1)  # حذف بعد اضافی برای سازگاری با BCEWithLogitsLoss






# تنظیم مسیر دیتاست
base_path = "Dataset/Brain Tumor MRI images"

# تعریف پردازش‌های اعمال‌شده روی تصاویر
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# لود دیتاست
full_dataset = datasets.ImageFolder(root=base_path, transform=transform)

# تقسیم دیتاست به train و validation
train_size = int(0.8 * len(full_dataset))  # 80% برای آموزش
valid_size = len(full_dataset) - train_size  # 20% برای اعتبارسنجی
train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size])

# ساخت DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)

# تعریف مدل
model = RESNet(num_classes=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# تعریف تابع هزینه و بهینه‌ساز
criterion = nn.BCEWithLogitsLoss()
# criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-3)

# حلقه آموزش
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loss, correct, total = 0, 0, 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)  # تبدیل لیبل به شکل (batch_size, 1)

        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        predictions = torch.sigmoid(outputs) > 0.5
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    
    train_acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss/len(train_loader):.4f}, Accuracy: {train_acc:.2f}%")

# ارزیابی مدل
model.eval()
valid_loss, correct, total = 0, 0, 0
with torch.no_grad():
    for images, labels in valid_loader:
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
        outputs = model(images)
        loss = criterion(outputs, labels)
        valid_loss += loss.item()
        predictions = torch.sigmoid(outputs) > 0.5
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

valid_acc = 100 * correct / total
print(f"Validation Loss: {valid_loss/len(valid_loader):.4f}, Accuracy: {valid_acc:.2f}%")

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np

def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predictions = (torch.sigmoid(outputs) > 0.5).cpu().numpy().astype(int).flatten()
            all_preds.extend(predictions)
            all_labels.extend(labels.cpu().numpy().astype(int).flatten())

    # نمایش Confusion Matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Non Tumor", "Tumor"], yticklabels=["Non Tumor", "Tumor"])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    # نمایش Classification Report
    print("Classification Report:\n", classification_report(all_labels, all_preds))

# اجرا برای داده‌های اعتبارسنجی
evaluate_model(model, valid_loader)

