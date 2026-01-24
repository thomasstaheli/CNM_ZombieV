#ifndef CUDA_PHYSICS_HPP
#define CUDA_PHYSICS_HPP

#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

class CudaPhysics {
private:
    // Pointeurs vers la mémoire du GPU (Device)
    float *d_posX, *d_posY;   // Position actuelle
    float *d_oldX, *d_oldY;   // Ancienne position
    float *d_accX, *d_accY;   // Accélération (incluant friction)
    
    // Buffers temporaires pour les calculs intermédiaires (Velocity)
    float *d_tempX, *d_tempY; 

    cublasHandle_t handle;    // Le contexte cuBLAS
    int m_capacity;           // Taille actuelle des buffers alloués

    void allocateBuffers(int count);
    void freeBuffers();

public:
    CudaPhysics();
    ~CudaPhysics();

    // Fonction principale qui fait l'aller-retour CPU -> GPU -> CPU
    void updatePositions(
        const std::vector<float>& in_posX, const std::vector<float>& in_posY,
        const std::vector<float>& in_oldX, const std::vector<float>& in_oldY,
        const std::vector<float>& in_accX, const std::vector<float>& in_accY,
        std::vector<float>& out_posX, std::vector<float>& out_posY,
        float dt, int count
    );
};

#endif // CUDA_PHYSICS_HPP