#!/bin/bash

# Arrêter le script si une commande échoue
set -e

# Se placer dans le dossier où se trouve ce script (la racine du projet)
cd "$(dirname "$0")"

echo "Compilation en cours..."
cmake --build build -j

echo "Lancement de ZombieV..."
# On lance l'exécutable depuis le dossier build
./build/ZombieV
