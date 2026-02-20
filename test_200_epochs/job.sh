#!/bin/bash
#SBATCH --job-name=resnet_cifar      # Nom du job
#SBATCH --output=resultat_%j.log     # Fichier de sortie (%j = ID du job)
#SBATCH --error=erreur_%j.err        # Fichier d'erreurs
#SBATCH --partition=ENSTA-l40s       # On utilise le nom exact vu dans sinfo
#SBATCH --gres=gpu:1                 # On réserve 1 GPU L40S
#SBATCH --cpus-per-task=4            # Nombre de coeurs CPU
#SBATCH --mem=16G                    # Mémoire RAM
#SBATCH --time=05:00:00              # Temps max (HH:MM:SS)

# Charger l'environnement (adapter selon l'installation du cluster)
module load anaconda3/2023.03        # Exemple : charger Anaconda
source activate mon_env_pytorch      # Si vous avez un environnement spécifique

# Lancer l'entraînement
python3 train_200_epochs.py