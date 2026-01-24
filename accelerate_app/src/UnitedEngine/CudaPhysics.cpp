#include "UnitedEngine/CudaPhysics.hpp"
#include <iostream>

CudaPhysics::CudaPhysics() : m_capacity(0), d_posX(nullptr) {
    // Initialisation de cuBLAS
    cublasCreate(&handle);
}

CudaPhysics::~CudaPhysics() {
    freeBuffers();
    cublasDestroy(handle);
}

void CudaPhysics::freeBuffers() {
    if (d_posX) {
        cudaFree(d_posX); cudaFree(d_posY);
        cudaFree(d_oldX); cudaFree(d_oldY);
        cudaFree(d_accX); cudaFree(d_accY);
        cudaFree(d_tempX); cudaFree(d_tempY);
        d_posX = nullptr;
    }
}

void CudaPhysics::allocateBuffers(int count) {
    if (count > m_capacity) {
        freeBuffers();
        m_capacity = count + 1000; // On alloue un peu plus pour éviter de réallouer trop souvent
        
        size_t size = m_capacity * sizeof(float);
        cudaMalloc(&d_posX, size); cudaMalloc(&d_posY, size);
        cudaMalloc(&d_oldX, size); cudaMalloc(&d_oldY, size);
        cudaMalloc(&d_accX, size); cudaMalloc(&d_accY, size);
        cudaMalloc(&d_tempX, size); cudaMalloc(&d_tempY, size);
    }
}

void CudaPhysics::updatePositions(
    const std::vector<float>& in_posX, const std::vector<float>& in_posY,
    const std::vector<float>& in_oldX, const std::vector<float>& in_oldY,
    const std::vector<float>& in_accX, const std::vector<float>& in_accY,
    std::vector<float>& out_posX, std::vector<float>& out_posY,
    float dt, int count) 
{
    // 1. Allocation dynamique si le nombre de zombies a augmenté
    allocateBuffers(count);

    size_t size = count * sizeof(float);

    // 2. Transfert CPU (Host) -> GPU (Device)
    // On copie les vecteurs contigus
    cudaMemcpy(d_posX, in_posX.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_posY, in_posY.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_oldX, in_oldX.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_oldY, in_oldY.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_accX, in_accX.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_accY, in_accY.data(), size, cudaMemcpyHostToDevice);

    // 3. CALCUL PHYSIQUE AVEC CUBLAS (SAXPY)
    // Formule Verlet : Pos = Pos + (Pos - OldPos) + Acc * dt
    // SAXPY fait : Y = alpha * X + Y

    float alpha_minus_1 = -1.0f;
    float alpha_1 = 1.0f;
    float alpha_dt = dt;

    // --- AXE X ---
    
    // A. Calcul de la vélocité (V = Pos - OldPos)
    // On copie Pos dans Temp
    cudaMemcpy(d_tempX, d_posX, size, cudaMemcpyDeviceToDevice); 
    // Temp = -1.0 * OldPos + Temp  =>  Temp = Pos - OldPos
    cublasSaxpy(handle, count, &alpha_minus_1, d_oldX, 1, d_tempX, 1);

    // B. Ajout de l'accélération (V = V + Acc * dt)
    // Temp = dt * Acc + Temp
    cublasSaxpy(handle, count, &alpha_dt, d_accX, 1, d_tempX, 1);

    // C. Mise à jour Position (Pos = Pos + V)
    // Pos = 1.0 * Temp + Pos
    cublasSaxpy(handle, count, &alpha_1, d_tempX, 1, d_posX, 1);


    // --- AXE Y (Même logique) ---
    cudaMemcpy(d_tempY, d_posY, size, cudaMemcpyDeviceToDevice); 
    cublasSaxpy(handle, count, &alpha_minus_1, d_oldY, 1, d_tempY, 1);
    cublasSaxpy(handle, count, &alpha_dt, d_accY, 1, d_tempY, 1);
    cublasSaxpy(handle, count, &alpha_1, d_tempY, 1, d_posY, 1);

    // 4. Transfert GPU (Device) -> CPU (Host)
    // On récupère les nouvelles positions
    cudaMemcpy(out_posX.data(), d_posX, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(out_posY.data(), d_posY, size, cudaMemcpyDeviceToHost);
}