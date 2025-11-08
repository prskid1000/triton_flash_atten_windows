# Building Flash Attention on Windows - Manual Instructions

This guide provides a single command to clone and build flash-attention on Windows.

## Prerequisites

- **Visual Studio 2022** (Community, Professional, or Enterprise) with C++ build tools
- **CUDA Toolkit 12.9** (or compatible version)
- **Python 3.12** (or compatible version)
- **PyTorch** installed in your virtual environment
- **Git** installed
- **Virtual environment** set up at `d:\omni\.venv` (or adjust paths accordingly)

## Single Command Build

Run this single command to clone the repository (if needed) and build flash-attention:

```cmd
cd D:\omni && if not exist "flash-attention" (git clone https://github.com/Dao-AILab/flash-attention.git flash-attention) && cd flash-attention && "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 && set MAX_JOBS=1 && set NVCC_THREADS=1 && set FLASH_ATTENTION_FORCE_BUILD=TRUE && set BUILD_TARGET=cuda && set DISTUTILS_USE_SDK=1 && set dist_dir=dist && set FLASH_ATTENTION_FORCE_CXX11_ABI=TRUE && set FLASH_ATTN_CUDA_ARCHS=80;120 && ..\.venv\Scripts\python.exe -m pip install "setuptools>=49.6.0" packaging wheel psutil && ..\.venv\Scripts\python.exe setup.py bdist_wheel --dist-dir=dist
```

**What this command does:**
1. Navigates to `D:\omni`
2. Clones the flash-attention repository if it doesn't exist
3. Navigates into the flash-attention directory
4. Initializes Visual Studio Developer Environment
5. Sets all required environment variables:
   - `MAX_JOBS=1`: Number of parallel build jobs (prevents OOM)
   - `NVCC_THREADS=1`: Threads per NVCC compilation (reduces memory)
   - `FLASH_ATTENTION_FORCE_BUILD=TRUE`: Force local build
   - `BUILD_TARGET=cuda`: Build for CUDA
   - `DISTUTILS_USE_SDK=1`: Use Windows SDK
   - `FLASH_ATTENTION_FORCE_CXX11_ABI=TRUE`: Force C++11 ABI
   - `FLASH_ATTN_CUDA_ARCHS=80;120`: CUDA architectures (sm_80, sm_120)
6. Installs build dependencies
7. Builds the wheel file

**Note:** If Visual Studio is installed in a different location, adjust the path:
- Professional: `C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat`
- Enterprise: `C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat`

## What Gets Compiled

This build compiles the **CUDA/CUTLASS extension** (`flash_attn_2_cuda`), which includes:
- All CUDA kernel files (`.cu` files) for forward and backward passes
- CUTLASS-based implementations for various attention operations

**Note:** Triton is **NOT compiled** during this build. Triton is a Python-based JIT compiler that:
- Is a separate package that must be installed via pip: `pip install triton`
- Provides alternative Python implementations (included in the package but not compiled)
- Compiles kernels at runtime when using Triton-based functions

If you want to use Triton-based implementations, install it separately:
```cmd
..\.venv\Scripts\python.exe -m pip install triton
```

## Verify the Build

After successful compilation, check for the wheel file:

```cmd
dir dist\*.whl
```

## Troubleshooting

### Error: "Cannot open include file: 'cstddef'"

**Solution:** Make sure you've run `vcvarsall.bat` to initialize the Visual Studio environment before building.

### Error: "nvcc fatal : '1 ': expected a number"

**Solution:** This was caused by trailing spaces in `NVCC_THREADS`. The setup.py has been fixed to strip whitespace. Ensure `NVCC_THREADS=1` (no spaces).

### Error: "NameError: name 'IS_ROCM' is not defined"

**Solution:** This has been fixed in setup.py. Ensure `BUILD_TARGET=cuda` is set correctly.

### Out of Memory (OOM) Errors

**Solution:** 
- Keep `MAX_JOBS=1` (single parallel job)
- Keep `NVCC_THREADS=1` (reduces memory per compilation)
- Close other memory-intensive applications

### Build Takes Too Long

**Solution:** You can try increasing `MAX_JOBS` if you have sufficient RAM:
- `set MAX_JOBS=2` (for systems with 64GB+ RAM)
- `set MAX_JOBS=4` (for systems with 128GB+ RAM, but monitor memory usage)

**Warning:** Increasing `MAX_JOBS` significantly increases memory usage. Each job with `NVCC_THREADS=1` still uses substantial memory.

## Customizing CUDA Architectures

To build for different CUDA architectures, modify `FLASH_ATTN_CUDA_ARCHS`:

```cmd
set FLASH_ATTN_CUDA_ARCHS=80;90;100;110;120
```

Common architectures:
- `80`: A100, A800
- `90`: H100
- `100`: Blackwell (requires CUDA 12.8+)
- `110`: Thor (requires CUDA 13.0+)
- `120`: Requires CUDA 12.8+

## Installing the Built Wheel

After successful build, install the wheel:

```cmd
..\.venv\Scripts\python.exe -m pip install dist\flash_attn-*.whl
```

## Notes

- The build process can take 30-60 minutes depending on your system
- Ensure you have at least 32GB of free RAM (64GB+ recommended)
- The Visual Studio environment must remain active in the same command prompt session
- All environment variables must be set before running `setup.py`

## Alternative: Using PowerShell

If you prefer PowerShell, use this command:

```powershell
cd D:\omni; if (-not (Test-Path "flash-attention")) { git clone https://github.com/Dao-AILab/flash-attention.git flash-attention }; cd flash-attention; cmd /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 && set MAX_JOBS=1 && set NVCC_THREADS=1 && set FLASH_ATTENTION_FORCE_BUILD=TRUE && set BUILD_TARGET=cuda && set DISTUTILS_USE_SDK=1 && set dist_dir=dist && set FLASH_ATTENTION_FORCE_CXX11_ABI=TRUE && set FLASH_ATTN_CUDA_ARCHS=80;120 && ..\.venv\Scripts\python.exe -m pip install "setuptools>=49.6.0" packaging wheel psutil && ..\.venv\Scripts\python.exe setup.py bdist_wheel --dist-dir=dist'
```

