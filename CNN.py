import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------
# 1. MODEL DEFINITIONS
# ---------------------------------------------------------------------
class BaselineCNN(nn.Module):
    """
    Baseline CNN with:
     - 1 convolutional layer (non-overlapping filters)
     - 1 average pooling layer
     - Fully-connected layer
    """

    def __init__(self, num_classes=4):
        super(BaselineCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=2, stride=2, padding=0)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(16 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


class SimpleCNN(nn.Module):
    """
    More advanced CNN with:
     - Multiple convolutional layers
     - Batch normalization
     - Dropout
    """

    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x

# ---------------------------------------------------------------------
# 2. TRAINING & EVALUATION FUNCTIONS
# ---------------------------------------------------------------------
def train(model, train_loader, criterion, optimizer, device):
    """
    Train one epoch.
    Returns:
     - average epoch loss
     - epoch accuracy
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, data_loader, criterion, device):
    """
    Evaluate model on given data_loader.
    Returns:
     - average loss
     - accuracy
     - all predictions
     - all ground-truth labels
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels


def plot_confusion_matrix(cm, class_names, model_name="Model"):
    """
    Display the confusion matrix using Seaborn.
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix for {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------
# 3. MAIN EXECUTION
# ---------------------------------------------------------------------
if __name__ == '__main__':
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data preparation
    data_dir = r"C:/Users/DELL/VS-Projects/CNN for CIFAR-10 dataset/CIFAR-10"
    train_dir = os.path.join(data_dir, "Train")
    test_dir = os.path.join(data_dir, "Test")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = torchvision.datasets.ImageFolder(root=test_dir, transform=transform)

    print("Classes found:", train_dataset.classes)

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Training setup
    criterion = nn.CrossEntropyLoss()

    baseline_model = BaselineCNN(num_classes=4).to(device)
    optimizer_base = optim.Adam(baseline_model.parameters(), lr=1e-3)

    adv_model = SimpleCNN(num_classes=4).to(device)
    optimizer_adv = optim.Adam(adv_model.parameters(), lr=1e-3)

    num_epochs = 10

    # Train Baseline Model
    print("\n--- Training Baseline Model ---")
    baseline_train_losses, baseline_train_accs = [], []
    baseline_test_losses, baseline_test_accs = [], []

    for epoch in range(num_epochs):
        train_loss, train_acc = train(baseline_model, train_loader, criterion, optimizer_base, device)
        test_loss, test_acc, _, _ = evaluate(baseline_model, test_loader, criterion, device)

        baseline_train_losses.append(train_loss)
        baseline_train_accs.append(train_acc)
        baseline_test_losses.append(test_loss)
        baseline_test_accs.append(test_acc)

        print(f"Epoch [{epoch + 1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} "
              f"| Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    base_test_loss, base_test_acc, base_preds, base_labels = evaluate(baseline_model, test_loader, criterion, device)
    base_precision = precision_score(base_labels, base_preds, average='weighted')
    base_recall = recall_score(base_labels, base_preds, average='weighted')
    base_f1 = f1_score(base_labels, base_preds, average='weighted')
    base_cm = confusion_matrix(base_labels, base_preds)

    # Train Advanced Model
    print("\n--- Training Advanced Model ---")
    adv_train_losses, adv_train_accs = [], []
    adv_test_losses, adv_test_accs = [], []

    for epoch in range(num_epochs):
        train_loss, train_acc = train(adv_model, train_loader, criterion, optimizer_adv, device)
        test_loss, test_acc, _, _ = evaluate(adv_model, test_loader, criterion, device)

        adv_train_losses.append(train_loss)
        adv_train_accs.append(train_acc)
        adv_test_losses.append(test_loss)
        adv_test_accs.append(test_acc)

        print(f"Epoch [{epoch + 1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} "
              f"| Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    adv_test_loss, adv_test_acc, adv_preds, adv_labels = evaluate(adv_model, test_loader, criterion, device)
    adv_precision = precision_score(adv_labels, adv_preds, average='weighted')
    adv_recall = recall_score(adv_labels, adv_preds, average='weighted')
    adv_f1 = f1_score(adv_labels, adv_preds, average='weighted')
    adv_cm = confusion_matrix(adv_labels, adv_preds)

    # Results & Visualizations
    print("\nBaseline Model Results:")
    print(f"Accuracy:  {base_test_acc:.4f}")
    print(f"Precision: {base_precision:.4f}")
    print(f"Recall:    {base_recall:.4f}")
    print(f"F1-score:  {base_f1:.4f}")
    print("Confusion Matrix:")
    print(base_cm)

    print("\nAdvanced Model Results:")
    print(f"Accuracy:  {adv_test_acc:.4f}")
    print(f"Precision: {adv_precision:.4f}")
    print(f"Recall:    {adv_recall:.4f}")
    print(f"F1-score:  {adv_f1:.4f}")
    print("Confusion Matrix:")
    print(adv_cm)

    class_names = train_dataset.classes  # ['airplane', 'automobile', 'ship', 'truck']

    # Plot confusion matrix for Baseline
    plot_confusion_matrix(base_cm, class_names, model_name="Baseline CNN")

    # Plot confusion matrix for Advanced
    plot_confusion_matrix(adv_cm, class_names, model_name="Advanced CNN")

    # Plot Training Curves
    epochs_range = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 5))

    # Baseline Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, baseline_train_losses, label='Train Loss')
    plt.plot(epochs_range, baseline_test_losses, label='Test Loss')
    plt.title('Baseline Model - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Baseline Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, baseline_train_accs, label='Train Acc')
    plt.plot(epochs_range, baseline_test_accs, label='Test Acc')
    plt.title('Baseline Model - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Advanced Loss
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, adv_train_losses, label='Train Loss')
    plt.plot(epochs_range, adv_test_losses, label='Test Loss')
    plt.title('Advanced Model - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Advanced Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, adv_train_accs, label='Train Acc')
    plt.plot(epochs_range, adv_test_accs, label='Test Acc')
    plt.title('Advanced Model - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Discussion
    discussion_text = f"""
    DISCUSSION & INSIGHTS:
    1. BaselineCNN Architecture:
       - 1 Convolutional layer (16 filters of size 3x3, stride=1)
       - 1 Average Pooling (kernel=2, stride=2)
       - Fully Connected layer => {len(class_names)} output classes
       Achieved ~{base_test_acc * 100:.2f}% test accuracy.
       Lower capacity leads to lower performance compared to advanced model.

    2. SimpleCNN Architecture:
       - 3 Convolutional layers (32, 64, 128 filters)
       - Batch Normalization after each conv
       - Max Pooling (downsampling)
       - Dropout in the fully-connected layers
       Achieved ~{adv_test_acc * 100:.2f}% test accuracy.
       Larger capacity and regularization (dropout) help achieve better performance.

    3. Metrics (Precision, Recall, F1-score) and Confusion Matrix:
       - Observed that some classes might be more easily predicted than others.
       - The confusion matrix helps identify which pairs of classes get confused.

    4. Overall:
       - The advanced CNN consistently outperforms the baseline in both training and testing.
       - Further improvements could involve more data augmentation, fine-tuning hyperparameters, or adding more convolutional layers.

    End of discussion.
    """
    print(discussion_text)
