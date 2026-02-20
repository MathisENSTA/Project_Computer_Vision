import torch
import torch.nn as nn
import torchvision.models as models

# 1. Créer le modèle de base
model = models.resnet18(num_classes=100)

# 2. ADAPTER L'ARCHITECTURE (indispensable pour CIFAR)
# On change la taille du noyau de 7x7 à 3x3 pour correspondre à votre sauvegarde
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
# Souvent, on retire aussi le maxpool pour CIFAR car l'image est déjà petite
model.maxpool = nn.Identity() 

# 3. Charger les poids
state_dict = torch.load('model_resnet18_final_300_epochs.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

model.eval()
print("Modèle chargé avec succès !")