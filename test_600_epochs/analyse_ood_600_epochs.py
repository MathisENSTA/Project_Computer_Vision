import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

# ==========================================
# 1. CONFIGURATION ET MODÈLE
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model():
    # Architecture exacte pour CIFAR-100
    model = models.resnet18(num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model

print("--- Chargement du modèle et des poids ---")
model = get_model()
state_dict = torch.load('model_resnet18_final_600_epochs.pth', map_location=device)
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

print("--- Préparation des données ---")
# In-Distribution (CIFAR-100)
testset_id = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset_id, batch_size=128, shuffle=False)

# Out-of-Distribution (SVHN)
testset_ood = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)
ood_loader = torch.utils.data.DataLoader(testset_ood, batch_size=128, shuffle=False)

# ==========================================
# 3. CALCUL DES SCORES OOD
# ==========================================
def get_scores(model, loader, method='msp'):
    all_scores = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            logits = model(images)
            
            if method == 'msp':
                # Plus la probabilité max est élevée, plus c'est "In-Distribution"
                probs = torch.softmax(logits, dim=1)
                score = torch.max(probs, dim=1)[0]
                
            elif method == 'logit':
                # Utilise simplement la valeur brute du logit max
                score = torch.max(logits, dim=1)[0]
                
            elif method == 'energy':
                # Energy score : log(sum(exp(logits)))
                # Un score élevé indique une donnée In-Distribution
                score = torch.logsumexp(logits, dim=1)
                
            elif method == 'mahalanobis':
                # Approximation par la norme L2 des logits (négative)
                score = -torch.norm(logits, p=2, dim=1)

            elif method == 'vim':
                # Version simplifiée combinant Logits et Énergie
                energy = torch.logsumexp(logits, dim=1)
                norm = torch.norm(logits, p=2, dim=1)
                score = energy + 0.05 * norm 
                
            all_scores.append(score.cpu().numpy())
            
    return np.concatenate(all_scores)

# ==========================================
# 4. ANALYSE ET GRAPHIQUES
# ==========================================
methods = ['msp', 'logit', 'energy', 'mahalanobis', 'vim']
all_results = {}

print("\nCalcul des scores pour chaque méthode...")
for m in methods:
    id_s = get_scores(model, testloader, method=m)
    ood_s = get_scores(model, ood_loader, method=m)
    all_results[m] = {'id': id_s, 'ood': ood_s}
    
    auroc = roc_auc_score([1]*len(id_s) + [0]*len(ood_s), np.concatenate([id_s, ood_s]))
    print(f"Méthode {m.upper():12} | AUROC: {auroc:.4f}")

# --- Tracé des Courbes ---
print("\nGénération des graphiques...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 1. Courbes ROC

for name, scores in all_results.items():
    y_true = np.array([1] * len(scores['id']) + [0] * len(scores['ood']))
    y_scores = np.concatenate([scores['id'], scores['ood']])
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    ax1.plot(fpr, tpr, label=f'{name.upper()} (AUROC={roc_auc_score(y_true, y_scores):.3f})')

ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax1.set_xlabel('Taux de Faux Positifs (FPR)')
ax1.set_ylabel('Taux de Vrais Positifs (TPR)')
ax1.set_title('Comparaison des Courbes ROC')
ax1.legend()
ax1.grid(alpha=0.3)

# 2. Histogramme (Comparaison ID vs OOD pour la méthode Energy)

method_to_plot = 'energy'
ax2.hist(all_results[method_to_plot]['id'], bins=50, alpha=0.6, label='ID (CIFAR-100)', density=True, color='blue')
ax2.hist(all_results[method_to_plot]['ood'], bins=50, alpha=0.6, label='OOD (SVHN)', density=True, color='red')
ax2.set_title(f'Distribution des scores : {method_to_plot.upper()}')
ax2.set_xlabel('Valeur du Score')
ax2.set_ylabel('Densité')
ax2.legend()

plt.tight_layout()
plt.savefig('analyse_ood_600_epochs.png', dpi=300)
print("Graphique sauvegardé sous 'analyse_ood_600_epochs.png'")