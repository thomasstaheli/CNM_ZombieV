cmake_minimum_required(VERSION 3.18)

set(PROJECT_NAME ZombieV)

project(${PROJECT_NAME} VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

find_package(OpenGL)
find_package(CUDAToolkit REQUIRED)

find_package(OpenMP REQUIRED) 

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake_modules" ${CMAKE_MODULE_PATH})
find_package(SFML 2 REQUIRED COMPONENTS audio graphics window system)

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

add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE "include" "lib")

target_link_libraries(${PROJECT_NAME} 
    sfml-system 
    sfml-window 
    sfml-graphics 
    sfml-audio
    CUDA::cublas
    CUDA::cudart
    OpenMP::OpenMP_CXX
)

if (UNIX)
   target_link_libraries(${PROJECT_NAME} pthread)
endif (UNIX)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

if(MSVC)
   foreach(lib ${SFML_LIBS})
      get_target_property(lib_path ${lib} LOCATION)
      file(COPY ${lib_path} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
   endforeach()
endif(MSVC)