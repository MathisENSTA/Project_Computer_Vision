import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.models import resnet18

# --- Configuration des données ---
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

# --- Modèle adapté pour CIFAR ---
def get_resnet18_cifar():
    model = resnet18(num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity() 
    return model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = get_resnet18_cifar().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

# MultiStepLR : 0.1 -> 0.01 (300) -> 0.001 (450)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300, 450], gamma=0.1)

# Listes pour mémoriser l'évolution
history = {'loss': [], 'acc': []}

def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    avg_loss = train_loss / len(trainloader)
    avg_acc = 100. * correct / total
    
    history['loss'].append(avg_loss)
    history['acc'].append(avg_acc)
    
    return avg_loss, avg_acc

# --- Boucle d'entraînement sur 600 époques ---
print("Début de l'entraînement (600 époques)...")
for epoch in range(600):
    loss, acc = train(epoch)
    scheduler.step()
    
    if epoch % 10 == 0 or epoch == 599:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}: Loss {loss:.4f} | Acc {acc:.2f}% | LR {current_lr}")
    
    # Sauvegarde de sécurité
    if (epoch + 1) % 100 == 0:
        torch.save(model.state_dict(), f'checkpoint_epoch_{epoch+1}.pth')

torch.save(model.state_dict(), 'model_resnet18_final_600_epochs.pth')
print("Entraînement terminé et modèle sauvegardé !")

# --- Génération des graphiques ---

plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(history['loss'], color='tab:red', label='Train Loss')
plt.axvline(x=300, color='gray', linestyle='--', label='LR Decay (x0.1)')
plt.axvline(x=450, color='gray', linestyle='--')
plt.title('Loss sur 600 Époques')
plt.xlabel('Époque')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(history['acc'], color='tab:blue', label='Train Accuracy')
plt.axvline(x=300, color='gray', linestyle='--')
plt.axvline(x=450, color='gray', linestyle='--')
plt.title('Accuracy sur 600 Époques')
plt.xlabel('Époque')
plt.ylabel('Précision (%)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('learning_curves_600_epochs.png')
print("Courbes sauvegardées : learning_curves_600_epochs.png")
plt.show()