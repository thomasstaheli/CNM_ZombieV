# ZombieV

2D top down zombie shooter game in C++ using SFML as graphics library and custom game engine (Ligths, Physics, Entity creation, etc...)

## Fork

If you want to trace the differents commits :

Github Repository : [https://github.com/thomasstaheli/CNM_ZombieV](https://github.com/thomasstaheli/CNM_ZombieV)

## Example
![Zombie](https://github.com/johnBuffer/ZombieV/blob/master/img/illustration.png)

### Video link

 - [Single player](https://www.youtube.com/watch?v=pj3m3Fu3i5A)
 - [Bots](https://www.youtube.com/watch?v=LflP2BUqJQc)
 - [Lights](https://www.youtube.com/watch?v=rCyaakRHUJ0)

## Run demo

In the release folder you will find binaries, each one corresponding to a scenario
*  Solo game
*  Game with bots
*  Lights demo
*  Bots + night

## Report

The report is not in this `README.md`, but in `rapport.pdf`.

## Building the Program

To run this program, you will need:

- A C++ compiler supporting C++17 or later
- CMake version 3.18 or newer
- The SFML library

### Original App

From the repository root:

```bash
cd original_app/build
# Builds and launches the game
../run.sh
```

### Accelerate App

From the repository root:

```bash
cd accelerate_app/build
# Builds and launches the game
../run.sh
```

### Quit the game

To quit the game, press Escape (ESC) or click the close button ("X") in the top-right corner of the window.

## How can I reproduce the test ?

After each run, a file `named measures_fps.csv` is generated, containing the FPS measurements. To visualize the data, you can use the Python script located at the root of the repository.

You will need, `python3` with the librairies `panda` and `matplotlib`.

```bash
python3 plot_mesure.py
```

If you want to reproduce the same graphic as us, you have to wait the wave 16 and then close the game.

### Architecture that we used

With the CUDA basics programm :

```
./deviceQuery
```

```
Detected 1 CUDA Capable device(s)

Device 0: "NVIDIA GeForce RTX 3050 Ti Laptop GPU"
  CUDA Driver Version / Runtime Version          13.1 / 12.3
  CUDA Capability Major/Minor version number:    8.6
  Total amount of global memory:                 4096 MBytes (4294443008 bytes)
  (020) Multiprocessors, (128) CUDA Cores/MP:    2560 CUDA Cores
  GPU Max Clock rate:                            1485 MHz (1.49 GHz)
  Memory Clock rate:                             5871 Mhz
  Memory Bus Width:                              128-bit
  L2 Cache Size:                                 2097152 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 
                                                 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total shared memory per multiprocessor:        102400 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  1536
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 1 copy engine(s)
  Run time limit on kernels:                     Yes
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Device supports Managed Memory:                Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      No
  Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >
```

With command :
```
~$ nvcc --version
```

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Wed_Nov_22_10:17:15_PST_2023
Cuda compilation tools, release 12.3, V12.3.107
Build cuda_12.3.r12.3/compiler.33567101_0
```
