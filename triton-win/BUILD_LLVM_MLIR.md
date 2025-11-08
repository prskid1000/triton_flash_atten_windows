# Building LLVM with MLIR on Windows

This guide provides step-by-step instructions to build LLVM with MLIR enabled on Windows, which is required for building Triton from source.

## Problem

Visual Studio's LLVM installation does **NOT** include MLIR, which is required by Triton.

The error you'll see when building Triton:
```
CMake Error: Could not find a package configuration file provided by "MLIR"
```

## Prerequisites

- **Visual Studio 2022** (Community, Professional, or Enterprise) with C++ build tools
- **CMake 3.20+** (included with Visual Studio 2022)
- **Ninja** build system
- **Python 3.8+** (for LLVM build scripts)
- **Git** installed
- **At least 64GB RAM** (recommended)
- **50GB+ free disk space**

## Step 1: Clone LLVM Repository

```cmd
cd D:\omni && if not exist "llvm-project" (git clone --depth 1 https://github.com/llvm/llvm-project.git llvm-project)
```

Or with PowerShell:
```powershell
cd D:\omni; if (-not (Test-Path "llvm-project")) { git clone --depth 1 https://github.com/llvm/llvm-project.git llvm-project }
```

**Note:** Using `--depth 1` clones only the latest commit, saving time and disk space.

## Step 2: Create Build Directory

```cmd
cd D:\omni\llvm-project && mkdir build 2>nul && cd build
```

## Step 3: Configure with CMake

**CRITICAL:** You **MUST** build LLVM for **x64 (64-bit)** architecture. Building for x86 (32-bit) will cause linker errors when building Triton. Always use the **x64 Visual Studio Developer Command Prompt**.

**Important:** 
- If you're using pyenv, you need to specify the Python executable directly to avoid detection issues.
- **Always initialize the x64 Visual Studio environment** before running CMake to ensure correct architecture.

### Recommended Method (Using x64 Developer Command Prompt):

**Step 3a: Initialize Visual Studio x64 Environment**

```cmd
"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
```

**Step 3b: Configure CMake**

```cmd
cd D:\omni\llvm-project\build && "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" -D Python3_EXECUTABLE=D:/omni/.venv/Scripts/python.exe -GNinja -S llvm -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=C:/llvm-install -DLLVM_ENABLE_PROJECTS="mlir;lld" -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" -DLLVM_BUILD_UTILS=ON -DLLVM_INSTALL_UTILS=ON -DMLIR_ENABLE_BINDINGS_PYTHON=OFF
```

### Single-Line Command (CMD with vcvarsall):

```cmd
cd D:\omni\llvm-project && cmd /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 && "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" -D Python3_EXECUTABLE=D:/omni/.venv/Scripts/python.exe -GNinja -S llvm -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=C:/llvm-install -DLLVM_ENABLE_PROJECTS="mlir;lld" -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" -DLLVM_BUILD_UTILS=ON -DLLVM_INSTALL_UTILS=ON -DMLIR_ENABLE_BINDINGS_PYTHON=OFF'
```

**What this does:**
- `-D Python3_EXECUTABLE`: Specifies Python path (avoids pyenv issues)
- `-GNinja`: Uses Ninja build system (faster than Visual Studio)
- `-DCMAKE_BUILD_TYPE=Release`: Release build (optimized)
- `-DCMAKE_INSTALL_PREFIX=C:/llvm-install`: Installation directory
- `-DLLVM_ENABLE_PROJECTS="mlir;lld"`: Enables MLIR and LLD projects
- `-DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU"`: Builds for host, NVIDIA, and AMD GPUs
- `-DLLVM_BUILD_UTILS=ON`: Builds LLVM utilities
- `-DLLVM_INSTALL_UTILS=ON`: Installs LLVM utilities
- `-DMLIR_ENABLE_BINDINGS_PYTHON=OFF`: Disables Python bindings (not needed)

**Note:** This configuration step takes 1-2 minutes.

## Step 4: Build and Install LLVM

**Important:** Ensure you're still in the x64 Visual Studio Developer Command Prompt environment.

```cmd
cd D:\omni\llvm-project\build && ninja install
```

Or with vcvarsall:

```cmd
cd D:\omni\llvm-project\build && cmd /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 && ninja install'
```

**Warning:** This takes **2-6 hours** depending on your CPU. The build runs in the background.

### Verify Architecture Before Building

Before starting the build, you can verify CMake detected x64 correctly:

```cmd
cd D:\omni\llvm-project\build && type CMakeCache.txt | findstr "CMAKE_SYSTEM_PROCESSOR"
```

You should see `CMAKE_SYSTEM_PROCESSOR:STRING=x64` or similar indicating x64 architecture.

### Monitor Build Progress

**Check if build is running:**
```cmd
tasklist | findstr ninja
```

**Check build log:**
```cmd
cd D:\omni\llvm-project\build && type .ninja_log | findstr /C:"^[0-9]" | find /C /V ""
```

**Check installation progress:**
```cmd
if exist "C:\llvm-install\lib\cmake\mlir\MLIRConfig.cmake" (echo MLIR installed!) else (echo Still building...)
```

**Check build directory size:**
```cmd
for /f "tokens=3" %a in ('dir /s /-c D:\omni\llvm-project\build ^| find "File(s)"') do @echo Build size: %a bytes
```

### Increase Build Parallelism (Optional)

If you have sufficient RAM (128GB+), you can speed up the build:

```cmd
cd D:\omni\llvm-project\build && ninja -j 4 install
```

**Warning:** Increasing parallelism significantly increases memory usage. Monitor your system.

## Step 5: Set Environment Variables

After the build completes, set these environment variables for Triton:

```cmd
set LLVM_SYSPATH=C:\llvm-install && set LLVM_INCLUDE_DIRS=C:\llvm-install\include && set LLVM_LIBRARY_DIR=C:\llvm-install\lib
```

**PowerShell:**
```powershell
$env:LLVM_SYSPATH = "C:\llvm-install"
$env:LLVM_INCLUDE_DIRS = "C:\llvm-install\include"
$env:LLVM_LIBRARY_DIR = "C:\llvm-install\lib"
```

**Permanent (User-level):**
```cmd
setx LLVM_SYSPATH "C:\llvm-install"
setx LLVM_INCLUDE_DIRS "C:\llvm-install\include"
setx LLVM_LIBRARY_DIR "C:\llvm-install\lib"
```

## Verify Installation

### Check MLIR Installation

Check that MLIR is properly installed:

```cmd
if exist "C:\llvm-install\lib\cmake\mlir\MLIRConfig.cmake" (echo ✅ MLIR installed successfully!) else (echo ❌ MLIR not found)
```

### Verify Architecture (CRITICAL)

**You MUST verify that LLVM was built for x64 architecture.** Building for x86 will cause linker errors when building Triton.

**PowerShell:**
```powershell
dumpbin /headers "C:\llvm-install\lib\LLVMCore.lib" | Select-String "machine"
```

**CMD:**
```cmd
dumpbin /headers "C:\llvm-install\lib\LLVMCore.lib" | findstr "machine"
```

**Expected output for x64:**
- Look for `8664 machine (x64)` or `8664 machine (AMD64)`
- **NOT** `14C machine (x86)` - this indicates x86 build (wrong architecture!)

**Quick verification script:**
```powershell
$lib = "C:\llvm-install\lib\LLVMCore.lib"
if (Test-Path $lib) {
    $result = dumpbin /headers $lib 2>$null | Select-String "8664 machine"
    if ($result) {
        Write-Host "✅ LLVM is x64 - Correct!" -ForegroundColor Green
    } else {
        Write-Host "❌ LLVM is NOT x64 - You need to rebuild for x64!" -ForegroundColor Red
    }
} else {
    Write-Host "❌ LLVM not found at $lib" -ForegroundColor Red
}
```

### Check Installation Size

```cmd
dir /s C:\llvm-install | find "File(s)"
```

## Complete Single-Line Build Command

If you want to do everything in one go (after cloning), **ensuring x64 build**:

```cmd
cd D:\omni\llvm-project && mkdir build 2>nul && cmd /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 && "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" -D Python3_EXECUTABLE=D:/omni/.venv/Scripts/python.exe -GNinja -S llvm -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=C:/llvm-install -DLLVM_ENABLE_PROJECTS="mlir;lld" -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" -DLLVM_BUILD_UTILS=ON -DLLVM_INSTALL_UTILS=ON -DMLIR_ENABLE_BINDINGS_PYTHON=OFF && cd build && ninja install'
```

**Note:** This command ensures x64 build by using `vcvarsall.bat x64` before CMake configuration.

## Troubleshooting

### Error: "Python3 not found" or "Cannot use the interpreter"

**Solution:**
- Temporarily remove pyenv from PATH before running CMake
- Use full path to venv Python: `-D Python3_EXECUTABLE=D:/omni/.venv/Scripts/python.exe`
- Ensure Python 3.8+ is available

**PowerShell workaround:**
```powershell
$oldPath = $env:PATH
$env:PATH = ($env:PATH -split ';' | Where-Object { $_ -notlike '*pyenv*' }) -join ';'
# Run cmake command here
$env:PATH = $oldPath
```

### Error: "CMake not found"

**Solution:**
- CMake is included with Visual Studio 2022
- Use full path: `"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"`
- Or install CMake separately from https://cmake.org/download/

### Build Takes Too Long

**Solution:**
- This is normal - LLVM build takes 2-6 hours
- You can increase parallelism if you have enough RAM:
  ```cmd
  ninja -j 4 install
  ```
- Monitor memory usage and adjust `-j` value accordingly
- Close other memory-intensive applications

### Out of Memory (OOM) Errors

**Solution:**
- Reduce parallelism: `ninja -j 1 install`
- Ensure you have at least 64GB RAM (128GB+ recommended)
- Close other applications
- Consider building only MLIR (not full LLVM) if possible

### Build Fails with Compiler Errors

**Solution:**
- Ensure Visual Studio 2022 C++ build tools are installed
- Run `vcvarsall.bat x64` before building
- Check that you have the latest Windows SDK

### Error: "library machine type 'x86' conflicts with target machine type 'x64'" when building Triton

**Problem:** LLVM was built for x86 (32-bit) but Triton requires x64 (64-bit).

**Solution:**
1. **Verify the problem:**
   ```cmd
   dumpbin /headers "C:\llvm-install\lib\LLVMCore.lib" | findstr "machine"
   ```
   If you see `14C machine (x86)`, LLVM was built for x86 (wrong).

2. **Clean the build directory:**
   ```cmd
   cd D:\omni\llvm-project\build
   del /q CMakeCache.txt
   rmdir /s /q CMakeFiles
   ```
   Or in PowerShell:
   ```powershell
   cd D:\omni\llvm-project\build
   Remove-Item -Recurse -Force CMakeCache.txt,CMakeFiles -ErrorAction SilentlyContinue
   ```

3. **Reconfigure for x64:**
   - **CRITICAL:** Use the x64 Visual Studio Developer Command Prompt
   - Run `vcvarsall.bat x64` before CMake
   - Re-run the CMake configuration command from Step 3

4. **Rebuild and reinstall:**
   ```cmd
   cd D:\omni\llvm-project\build && cmd /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 && ninja install'
   ```

5. **Verify x64 architecture:**
   ```cmd
   dumpbin /headers "C:\llvm-install\lib\LLVMCore.lib" | findstr "8664 machine"
   ```
   Should show `8664 machine (x64)` or `8664 machine (AMD64)`.

**Prevention:** Always use `vcvarsall.bat x64` before running CMake configuration to ensure x64 build.

## What Gets Built

This build creates:
- **LLVM Core** - Compiler infrastructure
- **MLIR** - Multi-Level Intermediate Representation (required for Triton)
- **LLD** - Linker
- **LLVM Utilities** - Tools like FileCheck, etc.

**Installation location:** `C:\llvm-install`
- `bin/` - Executables
- `include/` - Header files
- `lib/` - Libraries and CMake configs
- `lib/cmake/mlir/` - MLIR CMake configuration (required by Triton)

## Notes

- **One-time setup:** Once built, you can reuse `C:\llvm-install` for future Triton builds
- **Disk space:** Final installation is ~5-10 GB
- **Build time:** 2-6 hours depending on CPU
- **Reusable:** The built LLVM can be used for multiple Triton installations
- **Version:** This builds the latest LLVM from the main branch
- **CRITICAL - Architecture:** **ALWAYS build for x64 (64-bit)**. Building for x86 will cause linker errors when building Triton. Always use `vcvarsall.bat x64` before CMake configuration.
- **Verification:** After installation, always verify the architecture using `dumpbin /headers` to ensure x64 build (see Verify Installation section)

## Next Steps

After LLVM with MLIR is built:

1. Set environment variables (see Step 5)
2. Proceed with Triton installation (see `BUILD_TRITON.md`)
3. Verify MLIR is found by Triton's CMake configuration

## Alternative: Use Pre-built LLVM

If building takes too long, consider:
- Using WSL2 with pre-built Linux LLVM
- Looking for community-maintained Windows LLVM builds with MLIR
- Using Triton's pre-built wheels instead of building from source
