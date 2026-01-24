cmake_minimum_required(VERSION 3.18) # 3.18 minimum recommandé pour CMAKE_CUDA_ARCHITECTURES

set(PROJECT_NAME ZombieV)

# 1. Configuration du projet et des langages
project(${PROJECT_NAME} VERSION 1.0.0 LANGUAGES CXX CUDA)

# --- CORRECTION 1 : Standard C++ plus récent ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# --- CORRECTION 2 : Optimisation pour votre GPU ---
# "native" détecte automatiquement votre carte graphique pour optimiser le binaire
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

# 2. Recherche des paquets
find_package(OpenGL)
find_package(CUDAToolkit REQUIRED)

# --- CORRECTION 3 : Trouver OpenMP avant de l'utiliser ---
find_package(OpenMP REQUIRED) 

# Détecter SFML
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake_modules" ${CMAKE_MODULE_PATH})
find_package(SFML 2 REQUIRED COMPONENTS audio graphics window system)

# 3. Sources
file(GLOB source_files
    "src/*.cpp"
    "src/*.cu"             
    "src/Blur/*.cpp"
    "src/LightEngine/*.cpp"
    "src/Props/*.cpp"
    "src/System/*.cpp"
    "src/UnitedEngine/*.cpp"
    "src/Weapons/*.cpp"
)

set(SOURCES ${source_files})

# 4. Création de l'exécutable
add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE "include" "lib")

# 5. Liaison des librairies
target_link_libraries(${PROJECT_NAME} 
    sfml-system 
    sfml-window 
    sfml-graphics 
    sfml-audio
    CUDA::cublas    # Votre ajout correct pour cuBLAS
    CUDA::cudart    # Runtime CUDA
    OpenMP::OpenMP_CXX # Fonctionne maintenant grâce au find_package plus haut
)

# Gestion Linux/WSL (WSL est considéré comme UNIX)
if (UNIX)
   target_link_libraries(${PROJECT_NAME} pthread)
endif (UNIX)

# Copie des données
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# Ce bloc sera ignoré sur WSL (c'est pour Visual Studio sous Windows natif)
if(MSVC)
   foreach(lib ${SFML_LIBS})
      get_target_property(lib_path ${lib} LOCATION)
      file(COPY ${lib_path} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
   endforeach()
endif(MSVC)