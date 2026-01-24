#!/bin/bash
set -e

# Se placer dans le dossier racine du projet (là où se trouve original_app)
# Si le script est dans un sous-dossier, on remonte.
SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd "$SCRIPTPATH/.."
cd "accelerate_app"

echo "Configuration/Compilation en cours dans $(pwd)..."

# 1. Si le dossier build n'existe pas, on le crée
if [ ! -d "build" ]; then
    mkdir build
fi

# 2. On génère les fichiers de build (le fameux "cache")
# On ne le fait que si le cache n'existe pas déjà pour gagner du temps, 
# ou on peut le forcer ici.
if [ ! -f "build/CMakeCache.txt" ]; then
    echo "Génération du cache CMake..."
    cmake -B build -S .
fi

# 3. Maintenant on peut compiler
echo "Lancement de la compilation..."
cmake --build build -j$(nproc)

# 4. Lancement
if [ -f "./build/ZombieV" ]; then
    echo "Lancement de ZombieV..."
    ./build/ZombieV
else
    echo "Erreur : L'exécutable ./build/ZombieV est introuvable."
    exit 1
fi