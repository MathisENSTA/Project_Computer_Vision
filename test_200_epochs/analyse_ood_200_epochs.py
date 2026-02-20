import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from sklearn.metrics import roc_auc_score

# ==========================================
# 1. CONFIGURATION ET MODÈLE
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model():
    model = models.resnet18(num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model

print("Chargement du modèle...")
model = get_model()
state_dict = torch.load('model_resnet18_final_200_epochs.pth', map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# ==========================================
# 2. CHARGEMENT DES DATASETS (ID et OOD)
# ==========================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

print("Chargement des données ID (CIFAR-100)...")
testset_id = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset_id, batch_size=128, shuffle=False)

print("Chargement des données OOD (SVHN)...")
testset_ood = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)
ood_loader = torch.utils.data.DataLoader(testset_ood, batch_size=128, shuffle=False)

# ==========================================
# 3. FONCTIONS DE SCORING OOD
# ==========================================
def get_scores(model, loader, method='msp'):
    all_scores = []
    # Pour ViM et Mahalanobis, on aurait besoin d'extraire les features
    # Ici, nous utilisons une couche d'identité pour intercepter les features
    # On sauvegarde la couche fc originale
    original_fc = model.fc
    
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            
            # Récupération des Logits
            logits = model(images)
            
            if method == 'msp':
                # Max Softmax Probability (plus c'est haut, plus c'est ID)
                probs = torch.softmax(logits, dim=1)
                score = torch.max(probs, dim=1)[0]
                
            elif method == 'logit':
                # Max Logit Score
                score = torch.max(logits, dim=1)[0]
                
            elif method == 'energy':
                # Energy Score (T=1) : log(sum(exp(logits)))
                score = torch.logsumexp(logits, dim=1)
                
            elif method == 'mahalanobis':
                # Version simplifiée : distance à l'origine (négative car score élevé = ID)
                score = -torch.norm(logits, p=2, dim=1)

            elif method == 'vim':
                # Version simplifiée du Virtual Logit Matching
                # On combine l'énergie et la norme des logits
                energy = torch.logsumexp(logits, dim=1)
                norm = torch.norm(logits, p=2, dim=1)
                score = energy + 0.1 * norm # Approximation ViM
                
            all_scores.append(score.cpu().numpy())
            
    return np.concatenate(all_scores)

# ==========================================
# 4. COMPARAISON ET MÉTRIQUES (AUROC)
# ==========================================
def compute_auroc(id_scores, ood_scores):
    # En OOD, 1 = In-Distribution, 0 = Out-of-Distribution
    y_true = np.array([1] * len(id_scores) + [0] * len(ood_scores))
    y_scores = np.concatenate([id_scores, ood_scores])
    return roc_auc_score(y_true, y_scores)

methods = ['msp', 'logit', 'energy', 'mahalanobis', 'vim']
print("\n" + "="*40)
print(f"{'Méthode':<15} | {'AUROC Score':<12}")
print("-" * 40)

for m in methods:
    id_s = get_scores(model, testloader, method=m)
    ood_s = get_scores(model, ood_loader, method=m)
    auroc = compute_auroc(id_s, ood_s)
    print(f"{m.upper():<15} | {auroc:.4f}")

print("="*40)