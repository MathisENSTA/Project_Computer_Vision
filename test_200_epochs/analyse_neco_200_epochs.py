import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# 1. SETUP & MODÈLE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model():
    model = models.resnet18(num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model

model = get_model()
state_dict = torch.load('model_resnet18_final_200_epochs.pth', map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# 2. DATASETS
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

# On a besoin du trainset pour calculer les centres (prototypes) du Neural Collapse
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False)

testloader_id = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform),
    batch_size=128, shuffle=False)

testloader_ood = torch.utils.data.DataLoader(
    torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform),
    batch_size=128, shuffle=False)

# 3. EXTRACTION DES PROTOTYPES (NC1/NC2)
print("Calcul des prototypes de classes (Neural Collapse Centers)...")
feature_extractor = torch.nn.Sequential(*(list(model.children())[:-1]))
class_features = [[] for _ in range(100)]

with torch.no_grad():
    for images, labels in trainloader:
        feats = feature_extractor(images.to(device)).view(images.size(0), -1)
        for f, l in zip(feats, labels):
            class_features[l].append(f)

# Moyenne par classe pour obtenir les centres du simplexe
prototypes = torch.stack([torch.stack(cf).mean(0) for cf in class_features]) # [100, 512]
prototypes = prototypes.to(device)

# 4. MÉTHODE NECO
def get_neco_scores(model, loader):
    scores = []
    with torch.no_grad():
        for images, _ in loader:
            # Extraction des features de l'image de test
            feats = feature_extractor(images.to(device)).view(images.size(0), -1)
            
            # NECO Score : On mesure la projection ou la distance au simplexe
            # Ici on utilise la similarité cosinus maximale par rapport aux prototypes du NC
            # car le NC2 montre que les centres sont équiangulaires.
            norm_feats = torch.nn.functional.normalize(feats, p=2, dim=1)
            norm_protos = torch.nn.functional.normalize(prototypes, p=2, dim=1)
            
            # Produit scalaire pour obtenir la proximité avec la structure du simplexe
            cosine_similarities = torch.mm(norm_feats, norm_protos.t())
            score, _ = torch.max(cosine_similarities, dim=1)
            
            scores.append(score.cpu().numpy())
    return np.concatenate(scores)

# 5. ÉVALUATION
print("Évaluation de la méthode NECO...")
id_scores = get_neco_scores(model, testloader_id)
ood_scores = get_neco_scores(model, testloader_ood)

auroc = roc_auc_score([1]*len(id_scores) + [0]*len(ood_scores), 
                      np.concatenate([id_scores, ood_scores]))

print(f"\nRésultat NECO AUROC: {auroc:.4f}")

# 6. VISUALISATION
plt.figure(figsize=(10, 6))
plt.hist(id_scores, bins=50, alpha=0.5, label='ID (CIFAR-100)', density=True)
plt.hist(ood_scores, bins=50, alpha=0.5, label='OOD (SVHN)', density=True)
plt.title(f'NECO Detection (AUROC: {auroc:.4f})')
plt.xlabel('Similarité au Simplexe de Classes')
plt.legend()
plt.savefig('analyse_neco_200_epochs.png')
print("Graphique sauvegardé : analyse_neco_200_epochs.png")