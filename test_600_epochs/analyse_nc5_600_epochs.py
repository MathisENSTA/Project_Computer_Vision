import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np

# 1. SETUP & CHARGEMENT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model():
    model = models.resnet18(num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model

print("Chargement du modèle...")
model = get_model()
checkpoint = torch.load('model_resnet18_final_600_epochs.pth', map_location=device)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()

# 2. CHARGEMENT DES DONNÉES ID (CIFAR-100)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

# 3. EXTRACTION DES FEATURES ET DES LOGITS
print("Calcul des centres de classes et des prédictions...")
all_features = []
all_labels = []
all_preds_model = []

# Pour NC5, on a besoin des features (avant FC) et des sorties (après FC)
# On utilise un hook ou on sépare le forward
feature_extractor = torch.nn.Sequential(*(list(model.children())[:-1]))

with torch.no_grad():
    for images, labels in testloader:
        images = images.to(device)
        
        # Features (moyennées par le Global Average Pooling)
        feat = feature_extractor(images).view(images.size(0), -1)
        # Logits (sortie finale)
        logits = model(images)
        
        all_features.append(feat.cpu())
        all_labels.append(labels.cpu())
        all_preds_model.append(torch.argmax(logits, dim=1).cpu())

features = torch.cat(all_features, dim=0)
labels = torch.cat(all_labels, dim=0)
preds_model = torch.cat(all_preds_model, dim=0)

# 4. CALCUL DES CENTRES DE CLASSES (PROTOTYPES)
class_centers = []
for i in range(100):
    mask = (labels == i)
    if mask.any():
        class_centers.append(features[mask].mean(dim=0))
    else:
        class_centers.append(torch.zeros(features.shape[1]))

class_centers = torch.stack(class_centers) # Shape [100, 512]

# 5. DÉCISION PAR PLUS PROCHE VOISIN (NCC - Nearest Class Center)
# On calcule la distance entre chaque image et chaque centre de classe
distances = torch.cdist(features, class_centers) # Euclidienne
preds_ncc = torch.argmin(distances, dim=1)

# 6. ANALYSE NC5 : MESURE DE L'ACCORD
agreement = (preds_model == preds_ncc).float().mean().item() * 100
accuracy_model = (preds_model == labels).float().mean().item() * 100
accuracy_ncc = (preds_ncc == labels).float().mean().item() * 100

print("\n" + "="*40)
print(f"ANALYSE DU NEURAL COLLAPSE (NC5)")
print("-" * 40)
print(f"Précision du Modèle (Standard) : {accuracy_model:.2f}%")
print(f"Précision du Classifieur NCC    : {accuracy_ncc:.2f}%")
print(f"Accord (NC5)                   : {agreement:.2f}%")
print("-" * 40)

if agreement > 95:
    print("CONCLUSION : NC5 est validé. Le classifieur s'est effondré")
    print("sur une règle de décision par plus proche centre.")
else:
    print("CONCLUSION : NC5 est partiel. Le réseau utilise encore des")
    print("frontières de décision plus complexes que la simple distance.")
print("="*40)