import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt # Ajout pour les graphiques
from torchvision.models import resnet18

# --- Préparation des données ---
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

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# --- Modèle ---
def get_resnet18_cifar():
    model = resnet18(num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity() 
    return model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = get_resnet18_cifar().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# --- Listes pour stocker l'historique ---
history = {
    'train_loss': [],
    'train_acc': []
}

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
        
        # Calcul de l'accuracy
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    avg_loss = train_loss / len(trainloader)
    avg_acc = 100. * correct / total
    
    # Stockage
    history['train_loss'].append(avg_loss)
    history['train_acc'].append(avg_acc)
    
    print(f"Epoch {epoch}: Loss {avg_loss:.4f} | Acc {avg_acc:.2f}%")

# --- Boucle d'entraînement ---
print("Début de l'entraînement...")
for epoch in range(200):
    train(epoch)
    scheduler.step()

torch.save(model.state_dict(), 'model_resnet18_final_200_epochs.pth')
print("Modèle sauvegardé !")

# ==========================================
GÉNÉRATION DES FIGURES
# ==========================================

epochs_range = range(200)

plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history['train_loss'], label='Train Loss', color='red')
plt.title('Évolution de la Perte (Loss)')
plt.xlabel('Époques')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs_range, history['train_acc'], label='Train Accuracy', color='blue')
plt.title('Évolution de la Précision (Accuracy)')
plt.xlabel('Époques')
plt.ylabel('Précision (%)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('courbes_apprentissage.png')
print("Graphique sauvegardé sous 'courbes_apprentissage.png'")
plt.show()