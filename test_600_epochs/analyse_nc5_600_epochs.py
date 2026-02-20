import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. SETUP & CHARGEMENT DU MODÈLE
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model():
    # Architecture exacte utilisée pendant l'entraînement
    model = models.resnet18(num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model

print("Chargement du modèle")
model = get_model()
state_dict = torch.load('model_resnet18_final_600_epochs.pth', map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# ==========================================
# 2. CHARGEMENT DES DONNÉES (CIFAR-100)
# ==========================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

print("Préparation des données de test")
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

# ==========================================
# 3. EXTRACTION DES FEATURES ET PRÉDICTIONS
# ==========================================
print("Extraction des caractéristiques et calcul NC5")
all_features = []
all_labels = []
all_preds_model = []

# On isole l'extracteur de caractéristiques (tout sauf la dernière couche fc)
feature_extractor = torch.nn.Sequential(*(list(model.children())[:-1]))

with torch.no_grad():
    for images, labels in testloader:
        images = images.to(device)
        
        # 1. Caractéristiques (Features) en sortie du Global Average Pooling
        feat = feature_extractor(images).view(images.size(0), -1)
        # 2. Logits (Sortie du modèle complet)
        logits = model(images)
        
        all_features.append(feat.cpu())
        all_labels.append(labels.cpu())
        all_preds_model.append(torch.argmax(logits, dim=1).cpu())

features = torch.cat(all_features, dim=0)
labels = torch.cat(all_labels, dim=0)
preds_model = torch.cat(all_preds_model, dim=0)

# ==========================================
# 4. CALCUL DES CENTRES DE CLASSES (NCC)
# ==========================================
class_centers = []
for i in range(100):
    mask = (labels == i)
    if mask.any():
        class_centers.append(features[mask].mean(dim=0))
    else:
        # Cas rare où une classe n'est pas dans le batch de test
        class_centers.append(torch.zeros(features.shape[1]))

class_centers = torch.stack(class_centers) 

# Décision par Plus Proche Centre (Nearest Class Center)
# On calcule la distance euclidienne entre chaque feature et chaque centre
distances = torch.cdist(features, class_centers) 
preds_ncc = torch.argmin(distances, dim=1)

# ==========================================
# 5. MÉTRIQUES ET ACCORD
# ==========================================
accuracy_model = (preds_model == labels).float().mean().item() * 100
accuracy_ncc = (preds_ncc == labels).float().mean().item() * 100
agreement = (preds_model == preds_ncc).float().mean().item() * 100

print(f"\nPrécision Modèle Standard : {accuracy_model:.2f}%")
print(f"Précision NCC              : {accuracy_ncc:.2f}%")
print(f"Accord (NC5)              : {agreement:.2f}%")

# ==========================================
# 6. VISUALISATION DES RÉSULTATS
# ==========================================


def plot_nc5_results(acc_m, acc_n, agr):
    plt.figure(figsize=(10, 6))
    names = ['Modèle (FC Layer)', 'Plus Proche Centre (NCC)', 'Accord (NC5)']
    values = [acc_m, acc_n, agr]
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    bars = plt.bar(names, values, color=colors, alpha=0.8, edgecolor='black')
    
    plt.ylim(0, 115)
    plt.ylabel('Score (%)', fontsize=12)
    plt.title('Validation du Neural Collapse NC5 (CIFAR-100)', fontsize=14)
    
    # Ajout des étiquettes de texte sur les barres
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 2,
                 f'{height:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.savefig('analyse_nc5_600_epochs.png', dpi=300)
    print("\nGraphique sauvegardé sous 'analyse_nc5_600_epochs.png'")

plot_nc5_results(accuracy_model, accuracy_ncc, agreement)