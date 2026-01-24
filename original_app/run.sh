#!/bin/bash
set -e

# Détermine le dossier où se trouve le script et va à la racine du projet
# (en supposant que le script est dans un sous-dossier comme /scripts ou /bin)
SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd "$SCRIPTPATH/.."
cd "original_app"

echo "Compilation en cours dans $(pwd)..."
cmake --build build -j$(nproc)

if [ -f "./build/ZombieV" ]; then
    echo "Lancement de ZombieV..."
    ./build/ZombieV
else
    echo "Erreur : L'exécutable ./build/ZombieV est introuvable."
    exit 1
fi
