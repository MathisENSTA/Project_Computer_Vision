import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# 1. ARCHITECTURE DU MODÈLE (Exactement comme à l'entraînement)
def get_model():
    model = models.resnet18(num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model

# 2. CHARGEMENT DES DONNÉES
print("Chargement des données CIFAR-100...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

# 3. CHARGEMENT DU MODÈLE ET DES POIDS
print("Chargement du modèle sauvegardé...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model()
state_dict = torch.load('model_resnet18_final_200_epochs.pth', map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# 4. EXTRACTION DES FEATURES (AVANT LA COUCHE FC)
print("Extraction des caractéristiques...")
features_list = []
labels_list = []

# On remplace temporairement la couche finale par une Identité pour avoir les 512 features
original_fc = model.fc
model.fc = nn.Identity()

with torch.no_grad():
    for images, labels in testloader:
        images = images.to(device)
        output = model(images)
        features_list.append(output.cpu())
        labels_list.append(labels)

features = torch.cat(features_list, dim=0).numpy()
labels = torch.cat(labels_list, dim=0).numpy()

import torch.nn.functional as F

# Préparation des données
# On regroupe les features par classe
class_means = []
for i in range(100): # CIFAR-100
    mask = (labels == i)
    if mask.any():
        class_means.append(torch.mean(torch.from_numpy(features[mask]), dim=0))

class_means = torch.stack(class_means) # Shape: [C, d]
global_mean = torch.mean(torch.from_numpy(features), dim=0) # Moyenne globale

# --- CALCUL NC1 : Rapport de variance ---
# On compare la dispersion intra-classe à la dispersion des centres
centered_means = class_means - global_mean
sigma_b = torch.matmul(centered_means.T, centered_means) / 100 # Variance inter-classe

intra_class_var = 0
for i in range(100):
    mask = (labels == i)
    diff = torch.from_numpy(features[mask]) - class_means[i]
    intra_class_var += torch.matmul(diff.T, diff)
sigma_w = intra_class_var / len(features) # Variance intra-classe

nc1_ratio = torch.trace(sigma_w) / torch.trace(sigma_b)
print(f"NC1 - Rapport de variance : {nc1_ratio:.6f}")

# --- CALCUL NC2 : Équianularité ---
# On regarde si les angles entre les centres de classes sont tous identiques
normed_means = F.normalize(centered_means, p=2, dim=1)
cosine_sim = torch.matmul(normed_means, normed_means.T)
# On retire la diagonale (auto-corrélation)
off_diag = cosine_sim[~torch.eye(100, dtype=bool)]
print(f"NC2 - Cosine Similarity moyenne entre centres: {off_diag.mean():.6f}")

# --- CALCUL NC3 : Alignement avec les poids W ---
# Récupérer les poids de la dernière couche (fc)
W = original_fc.weight.data.cpu() # Shape [100, 512]
W_centered = W - torch.mean(W, dim=0)
W_normed = F.normalize(W_centered, p=2, dim=1)

# Corrélation entre les centres des features et les poids du classifieur
alignment = torch.trace(torch.matmul(W_normed, normed_means.T)) / 100
print(f"NC3 - Alignement Classifieur/Features : {alignment:.6f}")

# --- CALCUL NC4 : Décision par plus proche voisin ---
# On compare la prédiction du modèle avec la classe du centre le plus proche
with torch.no_grad():
    # Distance euclidienne entre chaque feature et chaque centre de classe
    dist = torch.cdist(torch.from_numpy(features), class_means)
    nearest_center_preds = torch.argmin(dist, dim=1)
    
    # On récupère les prédictions réelles du modèle (features + fc)
    model_preds = torch.argmax(original_fc(torch.from_numpy(features)), dim=1)
    
    agreement = (nearest_center_preds == model_preds).float().mean()
    print(f"NC4 - Accord Classifieur vs Proche Voisin: {agreement*100:.2f}%")



import matplotlib.pyplot as plt

# Simulation de l'évolution basée sur tes résultats et la théorie du NC
epochs_sim = np.array([100, 150, 200, 300, 500])
# On part de tes résultats actuels à l'époque 200 (0.64) et on projette la suite
nc1_sim = np.array([0.95, 0.82, 0.64, 0.30, 0.05]) 
nc3_sim = np.array([0.70, 0.85, 0.95, 0.98, 0.99])

fig, ax1 = plt.subplots(figsize=(10, 6))

# Axe pour NC1 (Log scale car le collapse est exponentiel)
color = 'tab:red'
ax1.set_xlabel('Epochs')
ax1.set_ylabel('NC1: Intra-class Variability (Log)', color=color)
ax1.semilogy(epochs_sim, nc1_sim, '--o', color=color, label='NC1 (Collapse)')
ax1.tick_params(axis='y', labelcolor=color)
ax1.axvline(x=200, color='gray', linestyle=':', label='Ton état actuel')

# Deuxième axe pour NC3
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('NC3: Classifier Alignment', color=color)
ax2.plot(epochs_sim, nc3_sim, '-s', color=color, label='NC3 (Alignment)')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Progression théorique du Neural Collapse post-convergence')
fig.tight_layout()
plt.savefig('evolution_theorique_nc_200_epochs.png')
print("Graphique d'évolution sauvegardé.")





# 5. VISUALISATION (T-SNE)
print("Calcul du T-SNE...")
# On sélectionne les 10 premières classes pour que le graphique soit lisible
num_classes_to_show = 10
mask = labels < num_classes_to_show
features_sub = features[mask]
labels_sub = labels[mask]

tsne = TSNE(n_components=2, init='pca', random_state=42)
embeddings = tsne.fit_transform(features_sub)

# 6. AFFICHAGE DU GRAPHIQUE
plt.figure(figsize=(12, 8))
scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels_sub, cmap='tab10', s=15, alpha=0.6)
plt.colorbar(scatter, ticks=range(num_classes_to_show))
plt.title(f"Visualisation du Neural Collapse (Classes 0-{num_classes_to_show-1})")
plt.xlabel("Dimension T-SNE 1")
plt.ylabel("Dimension T-SNE 2")
plt.grid(True, linestyle='--', alpha=0.5)

plt.savefig('analyse_nc_1_4_200_epochs.png', dpi=300)
print("Graphique sauvegardé sous : analyse_nc_1_4_200_epochs.png")