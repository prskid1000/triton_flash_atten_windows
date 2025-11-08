# Building Triton on Windows - Complete Guide

This guide provides complete step-by-step instructions to build Triton from source on Windows, including all necessary fixes, LLVM/MLIR setup, and environment configuration.

## Prerequisites

- **Visual Studio 2022** (Community, Professional, or Enterprise) with C++ build tools
- **CUDA Toolkit 12.9** (or compatible version) - must be installed
- **Python 3.12** (or compatible version)
- **CMake 3.20+** (included with Visual Studio 2022)
- **Ninja** build system
- **Git** installed
- **Virtual environment** set up at `d:\omni\.venv` (or adjust paths accordingly)
- **At least 64GB RAM** (recommended for LLVM build)
- **50GB+ free disk space** (for LLVM build)

## Overview

Building Triton on Windows requires:
1. **Building LLVM with MLIR for x64** (required, takes 2-6 hours) - See `BUILD_LLVM_MLIR.md`
2. **Applying Windows compatibility fixes** to Triton source code (27 files modified)
3. **Setting CUDA toolkit environment variables**
4. **Setting LLVM environment variables**
5. **Building Triton wheel**

**CRITICAL:** You need BOTH:
- **LLVM built for x64** (solves linker errors - see `BUILD_LLVM_MLIR.md`)
- **All Triton source code fixes** (solves compilation errors - see Step 2)

The LLVM x64 fix only solves architecture mismatch linker errors. All Windows compatibility fixes in Triton's source code are still required.

## Step 1: Clone Triton Repository

```cmd
cd D:\omni && if not exist "triton" (git clone https://github.com/triton-lang/triton.git triton)
```

Or with PowerShell:
```powershell
cd D:\omni; if (-not (Test-Path "triton")) { git clone https://github.com/triton-lang/triton.git triton }
```

## Step 2: Build LLVM with MLIR (x64)

**MUST be built for x64 architecture.** See `BUILD_LLVM_MLIR.md` for complete instructions.

Quick summary:
1. Clone LLVM: `git clone --depth 1 https://github.com/llvm/llvm-project.git llvm-project`
2. Configure with x64: Use `vcvarsall.bat x64` before CMake
3. Build and install: `ninja install` (takes 2-6 hours)
4. Verify x64: `dumpbin /headers "C:\llvm-install\lib\LLVMCore.lib" | findstr "8664 machine"`

## Step 3: Apply Windows Compatibility Fixes to Triton

**Total: 27 files modified, 419 insertions, 40 deletions**

These fixes are **REQUIRED** for Windows builds. They address:
- Windows-specific compilation issues
- LLVM/MLIR API compatibility for newer LLVM versions (19+)
- Unix-to-Windows function replacements

### Fix Category 1: setup.py Windows Compatibility (2 files)

#### Fix 1.1: `setup.py` - Skip download on Windows

**File:** `triton/setup.py`  
**Location:** Around line 339-344

**Change:** Add Windows check before attempting to download NVIDIA toolchain binaries:

```python
# After line 339 (system = platform.system()):
    # Skip download on Windows - pre-compiled binaries are not available for Windows.
    # Users should install NVIDIA CUDA Toolkit and set environment variables like
    # TRITON_PTXAS_PATH, TRITON_CUOBJDUMP_PATH, etc. to point to their CUDA installation.
    if system == "Windows":
        return
```

#### Fix 1.2: `setup.py` - Skip empty URLs

**File:** `triton/setup.py`  
**Location:** Around line 298-299

**Change:** Add check to skip download if URL is empty (Windows LLVM case):

```python
# Before:
        if not is_offline_build() and not input_defined and not input_compatible:

# After:
        # Skip download if URL is empty (e.g., Windows LLVM which requires user-provided LLVM)
        if not is_offline_build() and not input_defined and not input_compatible and p.url:
```

#### Fix 1.3: `setup.py` - Add CUDA include directory for Proton

**File:** `triton/setup.py`  
**Location:** Around line 445-448 (in `get_proton_cmake_args` method)

**Change:** Add CUDA include directory support:

```python
        cmake_args += ["-DCUPTI_INCLUDE_DIR=" + cupti_include_dir]
        # Add CUDA include directory (needed for cuda.h which is included by cupti.h)
        cuda_include_dir = get_env_with_keys(["TRITON_CUDART_PATH", "TRITON_CUDACRT_PATH"])
        if cuda_include_dir != "":
            cmake_args += ["-DCUDA_INCLUDE_DIR=" + cuda_include_dir]
```

#### Fix 1.4: `setup.py` - Disable tests on Windows

**File:** `triton/setup.py`  
**Location:** Around line 533-535 (in `build_extension` method)

**Change:** Automatically disable tests on Windows:

```python
        if is_offline_build():
            # unit test builds fetch googletests from GitHub
            cmake_args += ["-DTRITON_BUILD_UT=OFF"]
        # Disable tests on Windows to avoid LLVM architecture mismatch issues
        if platform.system() == "Windows":
            cmake_args += ["-DTRITON_BUILD_UT=OFF"]
```

### Fix Category 2: CMake Windows Compiler Flags (1 file)

#### Fix 2.1: `CMakeLists.txt` - Windows compiler flags

**File:** `triton/CMakeLists.txt`  
**Location:** Around line 80-86

**Change:** Remove `-fPIC` flag for Windows and add deprecation warning suppression:

```cmake
# Before:
if(NOT MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STDC_FORMAT_MACROS  -fPIC -std=gnu++17")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STDC_FORMAT_MACROS")
endif()

# After:
if(NOT MSVC AND NOT WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STDC_FORMAT_MACROS  -fPIC -std=gnu++17")
else()
  # Suppress MSVC deprecation warning for std::complex (STL4037)
  # This is needed because MLIR uses std::complex in ways that trigger the warning
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STDC_FORMAT_MACROS -D_SILENCE_NONFLOATING_COMPLEX_DEPRECATION_WARNING")
endif()
```

**Why:** 
- `-fPIC` is not supported on Windows MSVC
- MSVC 14.44+ deprecates `std::complex` for non-floating-point types, which MLIR uses

### Fix Category 3: LLVM/MLIR API Compatibility (6 files)

These fixes are required when building with LLVM 19+ or newer versions.

#### Fix 3.1: `python/src/llvm.cc` - UnsafeFPMath removal

**File:** `triton/python/src/llvm.cc`  
**Location:** Lines 1-13 (add Windows guards), line 26 (add include), lines 64-68 (conditional compilation)

**Changes:**

1. Add Windows macro guards at top:
```cpp
#ifdef _WIN32
// Prevent Windows macros from interfering with LLVM headers
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif
```

2. Add include after line 25:
```cpp
#include "llvm/Config/llvm-config.h"
```

3. Conditional compilation around line 64:
```cpp
  if (enable_fp_fusion)
    opt.AllowFPOpFusion = llvm::FPOpFusion::Fast;
  // UnsafeFPMath was removed in newer LLVM versions - FP math is now controlled via function attributes
  #if LLVM_VERSION_MAJOR < 19
  opt.UnsafeFPMath = false;
  #endif
  opt.NoInfsFPMath = false;
```

#### Fix 3.2: `lib/Dialect/TritonGPU/IR/Ops.cpp` - RegionBranchPoint API

**File:** `triton/lib/Dialect/TritonGPU/IR/Ops.cpp`  
**Location:** Around line 933-938

**Change:** Replace `getRegionOrNull()` with `getTerminatorPredecessorOrNull()`:

```cpp
// Before:
  assert(src.getRegionOrNull() == &getDefaultRegion());
  successors.push_back(RegionSuccessor(getResults()));

// After:
  // In newer MLIR, getRegionOrNull() was removed. Use getTerminatorPredecessorOrNull() instead.
  auto *terminator = src.getTerminatorPredecessorOrNull();
  assert(terminator && terminator->getParentRegion() == &getDefaultRegion());
  successors.emplace_back(nullptr, getResults());
```

#### Fix 3.3: `lib/Target/LLVMIR/LLVMDILocalVariable.cpp` - Operation destructor

**File:** `triton/lib/Target/LLVMIR/LLVMDILocalVariable.cpp`  
**Location:** Around line 247

**Change:** Pass `Operation` by pointer instead of by value:

```cpp
// Before:
  LLVM::DISubprogramAttr getDISubprogramAttr(Operation op) {
    auto funcOp = op.getParentOfType<LLVM::LLVMFuncOp>();

// After:
  LLVM::DISubprogramAttr getDISubprogramAttr(Operation *op) {
    auto funcOp = op->getParentOfType<LLVM::LLVMFuncOp>();
```

**Why:** In newer MLIR versions, `Operation` destructor is private, so operations cannot be passed by value.

#### Fix 3.4: `third_party/amd/include/Analysis/RangeAnalysis.h` - RegionSuccessor API

**File:** `triton/third_party/amd/include/Analysis/RangeAnalysis.h`  
**Location:** Around line 85-88

**Change:** Change parameter from `RegionBranchPoint` to `RegionSuccessor`:

```cpp
// Before:
  void visitRegionSuccessors(
      ProgramPoint *point, RegionBranchOpInterface branch,
      RegionBranchPoint successor,
      ArrayRef<dataflow::AbstractSparseLattice *> abstractLattices) override;

// After:
  void visitRegionSuccessors(
      ProgramPoint *point, RegionBranchOpInterface branch,
      RegionSuccessor successor,
      ArrayRef<dataflow::AbstractSparseLattice *> abstractLattices) override;
```

#### Fix 3.5: `third_party/amd/lib/Analysis/RangeAnalysis.cpp` - RegionSuccessor implementation

**File:** `triton/third_party/amd/lib/Analysis/RangeAnalysis.cpp`  
**Location:** Around line 703-706 and 790-803

**Changes:**

1. Update function signature (line ~703):
```cpp
// Before:
void TritonIntegerRangeAnalysis::visitRegionSuccessors(
    ProgramPoint *point, RegionBranchOpInterface branch,
    RegionBranchPoint successor,
    ArrayRef<dataflow::AbstractSparseLattice *> abstractLattices) {

// After:
void TritonIntegerRangeAnalysis::visitRegionSuccessors(
    ProgramPoint *point, RegionBranchOpInterface branch,
    RegionSuccessor successor,
    ArrayRef<dataflow::AbstractSparseLattice *> abstractLattices) {
```

2. Update `visitNonControlFlowArguments` calls (around line 787-800):
```cpp
// Before:
        visitNonControlFlowArguments(branch,
                                     RegionSuccessor(branch->getResults().slice(
                                         firstIndex, inputs.size())),
                                     lattices, firstIndex);

// After:
        visitNonControlFlowArguments(branch.getOperation(),
                                     RegionSuccessor(branch.getOperation(), branch->getResults().slice(
                                         firstIndex, inputs.size())),
                                     lattices, firstIndex);
```

**Why:** 
- `visitNonControlFlowArguments` requires an `Operation*` parameter (use `branch.getOperation()`)
- `RegionSuccessor` constructor for parent successors requires the operation itself, not `nullptr`

#### Fix 3.6: `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/LoadStoreOpToLLVM.cpp` - NVVM operation removal

**File:** `triton/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/LoadStoreOpToLLVM.cpp`  
**Location:** Around line 1810-1824

**Change:** Replace `NVVM::CpAsyncMBarrierArriveSharedOp` with inline PTX:

```cpp
// Before:
    NVVM::CpAsyncMBarrierArriveSharedOp::create(rewriter, loc,
                                                barrierMemObj.getBase(), noinc);

// After:
    // CpAsyncMBarrierArriveSharedOp was removed in newer MLIR versions
    // Use inline PTX assembly instead
    PTXBuilder ptxBuilder;
    std::string ptx = "cp.async.mbarrier.arrive.shared.b64";
    if (noinc) {
      ptx += ".noinc";
    }
    ptx += " [$0];";
    auto &arriveOp = *ptxBuilder.create(ptx);
    arriveOp({ptxBuilder.newOperand(barrierMemObj.getBase(), "r")},
             /*onlyAttachMLIRArgs=*/true);
    auto voidTy = void_ty(op->getContext());
    ptxBuilder.launch(rewriter, loc, voidTy);
```

**Why:** `NVVM::CpAsyncMBarrierArriveSharedOp` was removed in newer MLIR versions.

### Fix Category 4: Windows Macro Guards (9 files)

Windows headers define macros that conflict with LLVM code. Add guards at the very top of files that include LLVM headers.

#### Files requiring Windows macro guards:

1. `triton/python/src/ir.cc` - Lines 1-9
2. `triton/python/src/llvm.cc` - Lines 1-9
3. `triton/third_party/amd/python/triton_amd.cc` - Lines 1-9
4. `triton/third_party/nvidia/triton_nvidia.cc` - Lines 1-9
5. `triton/third_party/proton/Dialect/triton_proton.cc` - Lines 1-9
6. `triton/third_party/amd/include/TritonAMDGPUToLLVM/TargetUtils.h` - Lines 1-9
7. `triton/third_party/amd/lib/TritonAMDGPUToLLVM/TargetUtils.cpp` - Lines 1-9
8. `triton/third_party/amd/lib/TritonAMDGPUToLLVM/TargetInfo.cpp` - Lines 1-9
9. `triton/third_party/amd/lib/TritonAMDGPUToLLVM/TargetInfo.h` - Lines 1-9

**Add at the very top of each file (before any includes):**

```cpp
#ifdef _WIN32
// Prevent Windows macros from interfering with LLVM headers
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif
```

**Special case - `triton_amd.cc`:** Also reorder includes to put LLVM headers BEFORE `hipblas_instance.h`:

```cpp
// After Windows guards, include LLVM headers FIRST:
#include "llvm/TargetParser/TargetParser.h"
#include "llvm/ADT/SmallString.h"
// ... other LLVM headers ...
// THEN include Triton/AMD headers that might include windows.h:
#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "amd/include/hipblas_instance.h"  // This includes windows.h
```

### Fix Category 5: Windows Dynamic Library Loading (6 files)

Replace Unix `dlfcn.h` functions with Windows `LoadLibrary`/`GetProcAddress`.

#### Fix 5.1: `third_party/proton/csrc/include/Driver/Dispatch.h` - Main dispatch library

**File:** `triton/third_party/proton/csrc/include/Driver/Dispatch.h`  
**Location:** Lines 1-18 (includes), lines 72-110 (init function), lines 125-131 (exec function), lines 151-162 (getLibPath function)

**Changes:**

1. Add Windows macro guards and conditional includes (top of file):
```cpp
#ifdef _WIN32
// Prevent Windows macros from interfering with LLVM headers
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif
```

2. Replace `init` function (around line 72):
```cpp
  static void init(const char *name, void **lib) {
    if (*lib == nullptr) {
#ifdef _WIN32
      // Windows: use LoadLibrary
      auto dir = ExternLib::pathEnv == nullptr ? "" : getStrEnv(ExternLib::pathEnv);
      if (!dir.empty()) {
        auto fullPath = dir + "\\" + name;
        *lib = LoadLibraryA(fullPath.c_str());
      } else {
        *lib = LoadLibraryA(name);
      }
      if (*lib == nullptr) {
        DWORD error = GetLastError();
        throw std::runtime_error("Could not load `" + std::string(name) +
                                 "`. Error code: " + std::to_string(error));
      }
#else
      // Unix: use dlopen
      // ... original Unix code ...
#endif
    }
  }
```

3. Replace `exec` function (around line 125):
```cpp
      if (handler == nullptr) {
#ifdef _WIN32
        handler = reinterpret_cast<FnT>(GetProcAddress((HMODULE)ExternLib::lib, functionName));
#else
        handler = reinterpret_cast<FnT>(dlsym(ExternLib::lib, functionName));
#endif
```

4. Replace `getLibPath` function (around line 151):
```cpp
    if (ExternLib::lib != nullptr) {
#ifdef _WIN32
      // Windows: use GetModuleFileName
      char path[MAX_PATH];
      if (GetModuleFileNameA((HMODULE)ExternLib::lib, path, MAX_PATH) != 0) {
        return std::string(path);
      }
#else
      // Unix: use dladdr
      void *sym = dlsym(ExternLib::lib, ExternLib::symbolName);
      Dl_info info;
      if (dladdr(sym, &info)) {
        return info.dli_fname;
      }
#endif
    }
```

#### Fix 5.2: `third_party/nvidia/include/cublas_instance.h` - cuBLAS Windows support

**File:** `triton/third_party/nvidia/include/cublas_instance.h`  
**Location:** Lines 1-26 (guards and includes), line 55-58 (library name), lines 83-167 (loadCublasDylib), lines 167-177 (unloadCublasDylib)

**Changes:**

1. Add Windows macro guards at top (BEFORE any includes):
```cpp
#ifdef _WIN32
// Prevent Windows macros from interfering with LLVM headers
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif
```

2. Conditional includes:
```cpp
#include "cublas_types.h"
#include <stdexcept>
#include <string>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif
```

3. Conditional library name:
```cpp
#ifdef _WIN32
  static constexpr const char *name = "cublas64_12.dll";
#else
  static constexpr const char *name = "libcublas.so";
#endif
```

4. Replace `loadCublasDylib` function:
```cpp
  void loadCublasDylib() {
#ifdef _WIN32
    if (dylibHandle == nullptr) {
      dylibHandle = LoadLibraryA(name);
    }
    if (dylibHandle == nullptr) {
      DWORD error = GetLastError();
      throw std::runtime_error("Could not find `" + std::string(name) +
                               "`. Error code: " + std::to_string(error));
    }
    // Use GetProcAddress for all function pointers
    cublasLtCreate = (cublasLtCreate_t)GetProcAddress((HMODULE)dylibHandle, "cublasLtCreate");
    // ... similar for all other functions ...
#else
    // Original Unix code with dlopen/dlsym
    // ...
#endif
  }
```

5. Replace `unloadCublasDylib`:
```cpp
  void unloadCublasDylib() {
#ifdef _WIN32
    if (dylibHandle) {
      FreeLibrary((HMODULE)dylibHandle);
    }
#else
    dlclose(dylibHandle);
#endif
  }
```

#### Fix 5.3: `third_party/amd/include/hipblas_instance.h` - hipBLAS Windows support

**File:** `triton/third_party/amd/include/hipblas_instance.h`  
**Location:** Same pattern as cublas_instance.h

**Changes:** Apply the same Windows compatibility changes as `cublas_instance.h`:
- Windows macro guards at top
- Conditional includes (`windows.h` vs `dlfcn.h`)
- Library name: `"hipblas64.dll"` on Windows, `"libhipblaslt.so"` on Unix
- Replace `dlopen`/`dlsym`/`dlclose` with `LoadLibraryA`/`GetProcAddress`/`FreeLibrary`

#### Fix 5.4: `third_party/proton/csrc/lib/Driver/GPU/HipApi.cpp` - HIP API Windows

**File:** `triton/third_party/proton/csrc/lib/Driver/GPU/HipApi.cpp`  
**Location:** Around lines 87-95 and 104-112

**Changes:** Replace `dlsym` with `GetProcAddress`:

```cpp
  if (func == nullptr) {
#ifdef _WIN32
    func = reinterpret_cast<hipKernelNameRef_t>(
        GetProcAddress((HMODULE)ExternLibHip::lib, "hipKernelNameRef"));
#else
    func = reinterpret_cast<hipKernelNameRef_t>(
        dlsym(ExternLibHip::lib, "hipKernelNameRef"));
#endif
  }
```

Apply same pattern to `getKernelNameRefByPtr` function.

#### Fix 5.5: `third_party/proton/csrc/lib/Driver/GPU/HsaApi.cpp` - HSA API Windows

**File:** `triton/third_party/proton/csrc/lib/Driver/GPU/HsaApi.cpp`  
**Location:** Around lines 25-35

**Change:** Replace `dlsym` with `GetProcAddress`:

```cpp
  if (func == nullptr) {
#ifdef _WIN32
    func = reinterpret_cast<hsa_iterate_agents_t>(
        GetProcAddress((HMODULE)ExternLibHsa::lib, "hsa_iterate_agents"));
#else
    func = reinterpret_cast<hsa_iterate_agents_t>(
        dlsym(ExternLibHsa::lib, "hsa_iterate_agents"));
#endif
  }
```

#### Fix 5.6: `third_party/proton/csrc/lib/Driver/GPU/RoctracerApi.cpp` - Roctracer Windows

**File:** `triton/third_party/proton/csrc/lib/Driver/GPU/RoctracerApi.cpp`  
**Location:** Around lines 16-25, 28-37, 40-49

**Changes:** Replace `dlsym` with `GetProcAddress` in three functions:
- `start()` - `roctracer_start`
- `stop()` - `roctracer_stop`
- `getOpString()` - `roctracer_op_string`

```cpp
  if (func == nullptr) {
#ifdef _WIN32
    func = reinterpret_cast<roctracer_start_t>(
        GetProcAddress((HMODULE)ExternLibRoctracer::lib, "roctracer_start"));
#else
    func = reinterpret_cast<roctracer_start_t>(
        dlsym(ExternLibRoctracer::lib, "roctracer_start"));
#endif
  }
```

### Fix Category 6: Proton Windows Function Compatibility (4 files)

#### Fix 6.1: `third_party/proton/csrc/lib/Driver/GPU/NvtxApi.cpp` - setenv replacement

**File:** `triton/third_party/proton/csrc/lib/Driver/GPU/NvtxApi.cpp`  
**Location:** Lines 3-8 (includes), lines 25-40 (enable/disable functions)

**Changes:**

1. Add Windows header:
```cpp
#include <cstdint>
#include <cstdlib>
#ifdef _WIN32
#include <stdlib.h>
#endif
```

2. Replace `setenv`/`unsetenv`:
```cpp
void enable() {
  const std::string cuptiLibPath = Dispatch<cupti::ExternLibCupti>::getLibPath();
  if (!cuptiLibPath.empty()) {
#ifdef _WIN32
    _putenv_s("NVTX_INJECTION64_PATH", cuptiLibPath.c_str());
#else
    setenv("NVTX_INJECTION64_PATH", cuptiLibPath.c_str(), 1);
#endif
  }
}

void disable() {
#ifdef _WIN32
    _putenv_s("NVTX_INJECTION64_PATH", "");
#else
    unsetenv("NVTX_INJECTION64_PATH");
#endif
}
```

#### Fix 6.2: `third_party/proton/csrc/lib/Profiler/Cupti/CuptiProfiler.cpp` - aligned_alloc replacement

**File:** `triton/third_party/proton/csrc/lib/Profiler/Cupti/CuptiProfiler.cpp`  
**Location:** Around lines 238-248 and 274-280

**Changes:**

1. Replace `aligned_alloc`:
```cpp
void CuptiProfiler::CuptiProfilerPimpl::allocBuffer(uint8_t **buffer,
                                                    size_t *bufferSize,
                                                    size_t *maxNumRecords) {
#ifdef _WIN32
  *buffer = static_cast<uint8_t *>(_aligned_malloc(BufferSize, AlignSize));
#else
  *buffer = static_cast<uint8_t *>(aligned_alloc(AlignSize, BufferSize));
#endif
```

2. Replace `free` with `_aligned_free`:
```cpp
#ifdef _WIN32
  _aligned_free(buffer);
#else
  std::free(buffer);
#endif
```

#### Fix 6.3: `third_party/proton/csrc/lib/Profiler/RocTracer/RoctracerProfiler.cpp` - cxxabi.h conditional

**File:** `triton/third_party/proton/csrc/lib/Profiler/RocTracer/RoctracerProfiler.cpp`  
**Location:** Around lines 18-24

**Change:** Conditionally include Unix-only headers:

```cpp
#include <mutex>
#include <tuple>

#ifndef _WIN32
#include <cxxabi.h>
#include <unistd.h>
#endif
```

#### Fix 6.4: `third_party/proton/csrc/include/Driver/GPU/HsaApi.h` - __attribute__ workaround

**File:** `triton/third_party/proton/csrc/include/Driver/GPU/HsaApi.h`  
**Location:** Lines 1-12

**Change:** Define `__attribute__` as empty macro for Windows:

```cpp
#ifndef PROTON_DRIVER_GPU_HSA_API_H_
#define PROTON_DRIVER_GPU_HSA_API_H_

#ifdef _WIN32
// MSVC doesn't support __attribute__, define it as empty for HSA headers
#ifndef __attribute__
#define __attribute__(x)
#endif
#ifndef __inline__
#define __inline__ inline
#endif
#endif

#include "Device.h"
#include "hsa/hsa_ext_amd.h"
```

### Fix Category 7: Proton CMake CUDA Include (1 file)

#### Fix 7.1: `third_party/proton/CMakeLists.txt` - CUDA include directory

**File:** `triton/third_party/proton/CMakeLists.txt`  
**Location:** Around lines 36-50 (in `add_proton_library` function)

**Change:** Add CUDA include directory support:

```cmake
  target_include_directories(${name}
    PRIVATE
      "${CUPTI_INCLUDE_DIR}"
      "${JSON_INCLUDE_DIR}"
      "${PROTON_COMMON_DIR}/include"
      "${PROTON_SRC_DIR}/include"
  )
  
  # Add CUDA include directory if provided (needed for cuda.h which is included by cupti.h)
  if(CUDA_INCLUDE_DIR)
    target_include_directories(${name}
      SYSTEM PRIVATE
        "${CUDA_INCLUDE_DIR}"
    )
  endif()
```

## Step 4: Set Environment Variables

### 4.1: Set CUDA Toolkit Variables

```cmd
set TRITON_PTXAS_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\ptxas.exe
set TRITON_PTXAS_BLACKWELL_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\ptxas.exe
set TRITON_CUOBJDUMP_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\cuobjdump.exe
set TRITON_NVDISASM_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvdisasm.exe
set TRITON_CUDACRT_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\include
set TRITON_CUDART_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\include
set TRITON_CUPTI_INCLUDE_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\extras\CUPTI\include
set TRITON_CUPTI_LIB_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\extras\CUPTI\lib64
```

**Note:** Adjust paths if your CUDA version differs. Check your installation:
```cmd
dir "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
```

### 4.2: Set LLVM Variables

After LLVM build completes (must be x64):

```cmd
set LLVM_SYSPATH=C:\llvm-install
set LLVM_INCLUDE_DIRS=C:\llvm-install\include
set LLVM_LIBRARY_DIR=C:\llvm-install\lib
```

## Step 5: Build Triton Wheel

Once LLVM is built for x64 and all fixes are applied:

### Single-Line Build Command

```cmd
cd D:\omni && cmd /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 && set MAX_JOBS=1 && set TRITON_PTXAS_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\ptxas.exe && set TRITON_PTXAS_BLACKWELL_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\ptxas.exe && set TRITON_CUOBJDUMP_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\cuobjdump.exe && set TRITON_NVDISASM_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvdisasm.exe && set TRITON_CUDACRT_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\include && set TRITON_CUDART_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\include && set TRITON_CUPTI_INCLUDE_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\extras\CUPTI\include && set TRITON_CUPTI_LIB_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\extras\CUPTI\lib64 && set LLVM_SYSPATH=C:\llvm-install && set LLVM_INCLUDE_DIRS=C:\llvm-install\include && set LLVM_LIBRARY_DIR=C:\llvm-install\lib && cd triton && ..\.venv\Scripts\python.exe -m pip wheel . --no-deps -w dist'
```

**PowerShell Alternative:**

```powershell
cd D:\omni; cmd /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 && set MAX_JOBS=1 && set TRITON_PTXAS_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\ptxas.exe && set TRITON_PTXAS_BLACKWELL_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\ptxas.exe && set TRITON_CUOBJDUMP_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\cuobjdump.exe && set TRITON_NVDISASM_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvdisasm.exe && set TRITON_CUDACRT_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\include && set TRITON_CUDART_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\include && set TRITON_CUPTI_INCLUDE_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\extras\CUPTI\include && set TRITON_CUPTI_LIB_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\extras\CUPTI\lib64 && set LLVM_SYSPATH=C:\llvm-install && set LLVM_INCLUDE_DIRS=C:\llvm-install\include && set LLVM_LIBRARY_DIR=C:\llvm-install\lib && cd triton && ..\.venv\Scripts\python.exe -m pip wheel . --no-deps -w dist'
```

The wheel file will be created in `triton/dist/triton-*.whl`.

## Verify Installation

After building, verify the wheel was created:

```cmd
dir D:\omni\triton\dist\*.whl
```

Install the wheel:

```cmd
cd D:\omni\triton && ..\.venv\Scripts\python.exe -m pip install dist\triton-*.whl
```

Test installation:

```cmd
..\.venv\Scripts\python.exe -c "import triton; print(f'Triton version: {triton.__version__}')"
```

## Troubleshooting

### Error: "KeyError: 'Windows'"

**Solution:** Apply Fix 1.1 in Step 3. The `download_and_copy` function needs to handle Windows.

### Error: "ValueError: unknown url type: ''"

**Solution:** Apply Fix 1.2 in Step 3. The `get_thirdparty_packages` function needs to skip empty URLs.

### Error: "Could NOT find MLIR"

**Solution:** 
- Ensure LLVM build completed successfully for x64
- Check that `C:\llvm-install\lib\cmake\mlir\MLIRConfig.cmake` exists
- Verify `LLVM_SYSPATH`, `LLVM_INCLUDE_DIRS`, and `LLVM_LIBRARY_DIR` are set correctly
- Verify LLVM is x64: `dumpbin /headers "C:\llvm-install\lib\LLVMCore.lib" | findstr "8664 machine"`

### Error: "Cannot find ptxas.exe"

**Solution:**
- Verify CUDA Toolkit is installed
- Check that `TRITON_PTXAS_PATH` points to the correct location
- Adjust paths if CUDA version differs from v12.9

### Error: "clang++: error: unsupported option '-fPIC'"

**Solution:** Apply Fix 2.1 in Step 3. The `-fPIC` flag is not supported on Windows MSVC.

### Error: "'complex' is deprecated: warning STL4037"

**Solution:** Fix 2.1 in Step 3 already includes the suppression macro `_SILENCE_NONFLOATING_COMPLEX_DEPRECATION_WARNING`.

### Error: "UnsafeFPMath is not a member of llvm::TargetOptions"

**Solution:** Apply Fix 3.1 in Step 3. This occurs when building with LLVM 19+ where `UnsafeFPMath` was removed.

### Error: "'getRegionOrNull' is not a member of 'mlir::RegionBranchPoint'"

**Solution:** Apply Fix 3.2 in Step 3. Newer MLIR versions removed `getRegionOrNull()` in favor of `getTerminatorPredecessorOrNull()`.

### Error: "'visitRegionSuccessors': method with override specifier 'override' did not override any base class methods"

**Solution:** Apply Fix 3.4 and 3.5 in Step 3. The DataFlow analysis API changed from `RegionBranchPoint` to `RegionSuccessor` parameter.

### Error: "'visitNonControlFlowArguments': function does not take 3 arguments"

**Solution:** Apply Fix 3.5 in Step 3. The `visitNonControlFlowArguments` method requires an `Operation*` parameter (use `branch.getOperation()`), and `RegionSuccessor` constructor for parent successors requires the operation itself.

### Error: "'mlir::Operation::~Operation': cannot access private member"

**Solution:** Apply Fix 3.3 in Step 3. In newer MLIR versions, `Operation` destructor is private, so operations must be passed by pointer, not by value.

### Error: "Cannot open include file: 'dlfcn.h': No such file or directory"

**Solution:** Apply Fix Category 5 in Step 3. `dlfcn.h` is Unix-only. Windows requires `windows.h` and `LoadLibrary`/`GetProcAddress` instead of `dlopen`/`dlsym`.

### Error: "'(': illegal token on right side of '::'" or "use of undefined type 'llvm::llvm::SMLoc'"

**Solution:** Apply Fix Category 4 in Step 3. Windows macros conflict with LLVM code. Add `NOMINMAX` and `WIN32_LEAN_AND_MEAN` guards at the very top of all files that include LLVM headers.

### Error: "syntax error: 'constant'" in `llvm/TargetParser/TargetParser.h`

**Solution:** Apply Fix Category 4 in Step 3, especially for `triton_amd.cc`. Reorder includes to put LLVM headers (especially `llvm/TargetParser/TargetParser.h`) BEFORE any headers that include `windows.h` (such as `hipblas_instance.h`).

### Error: "'getIsaVersion': is not a member of 'llvm::AMDGPU'"

**Solution:** This is usually caused by Windows macro conflicts. Apply Fix Category 4 to all files that include `TargetParser.h`, and ensure include order in `triton_amd.cc` puts LLVM headers before Windows.h includes.

### Error: "library machine type 'x86' conflicts with target machine type 'x64'"

**Solution:** LLVM was built for x86 but Triton requires x64. Rebuild LLVM for x64 (see `BUILD_LLVM_MLIR.md`). This fix does NOT replace the Triton source code fixes - you still need all fixes in Step 3.

### Error: "Cannot open include file: 'cuda.h': No such file or directory"

**Solution:** Apply Fix 1.3 and 7.1 in Step 3. Ensure `TRITON_CUDART_PATH` or `TRITON_CUDACRT_PATH` is set to the CUDA include directory.

### Error: "error C3861: 'setenv': identifier not found"

**Solution:** Apply Fix 6.1 in Step 3. Windows uses `_putenv_s` instead of `setenv`.

### Error: "'aligned_alloc': identifier not found"

**Solution:** Apply Fix 6.2 in Step 3. Windows uses `_aligned_malloc` instead of `aligned_alloc`.

### Error: "Cannot open include file: 'cxxabi.h'"

**Solution:** Apply Fix 6.3 in Step 3. `cxxabi.h` is Unix-only and should be conditionally included.

### Error: "syntax error: missing ';' before identifier '__attribute__'"

**Solution:** Apply Fix 6.4 in Step 3. MSVC doesn't support `__attribute__` - define it as empty macro.

### Error: "Cannot open include file: 'stddef.h': No such file or directory"

**Solution:** Visual Studio environment variables aren't properly inherited. Use `vcvarsall.bat x64` before building, or use the single-line command in Step 5 which includes it.

### Error: "'CpAsyncMBarrierArriveSharedOp': the symbol to the left of a '::' must be a type"

**Solution:** Apply Fix 3.6 in Step 3. `NVVM::CpAsyncMBarrierArriveSharedOp` was removed in newer MLIR versions. Use inline PTX assembly instead.

## Summary of All Changes

**Total: 27 files modified, 419 insertions(+), 40 deletions(-)**

### Files Modified:

1. `setup.py` - 4 fixes (Windows download skip, empty URL check, CUDA include, test disabling)
2. `CMakeLists.txt` - 1 fix (Windows compiler flags)
3. `python/src/llvm.cc` - 1 fix (UnsafeFPMath conditional, Windows guards)
4. `python/src/ir.cc` - 1 fix (Windows macro guards)
5. `lib/Dialect/TritonGPU/IR/Ops.cpp` - 1 fix (RegionBranchPoint API)
6. `lib/Target/LLVMIR/LLVMDILocalVariable.cpp` - 1 fix (Operation pointer)
7. `third_party/amd/include/Analysis/RangeAnalysis.h` - 1 fix (RegionSuccessor signature)
8. `third_party/amd/lib/Analysis/RangeAnalysis.cpp` - 1 fix (RegionSuccessor implementation)
9. `third_party/amd/include/TritonAMDGPUToLLVM/TargetUtils.h` - 1 fix (Windows guards)
10. `third_party/amd/lib/TritonAMDGPUToLLVM/TargetUtils.cpp` - 1 fix (Windows guards)
11. `third_party/amd/lib/TritonAMDGPUToLLVM/TargetInfo.cpp` - 1 fix (Windows guards)
12. `third_party/amd/lib/TritonAMDGPUToLLVM/TargetInfo.h` - 1 fix (Windows guards)
13. `third_party/amd/python/triton_amd.cc` - 1 fix (Windows guards + include reorder)
14. `third_party/amd/include/hipblas_instance.h` - 1 fix (Windows LoadLibrary)
15. `third_party/nvidia/include/cublas_instance.h` - 1 fix (Windows LoadLibrary)
16. `third_party/nvidia/triton_nvidia.cc` - 1 fix (Windows guards)
17. `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/LoadStoreOpToLLVM.cpp` - 1 fix (NVVM operation removal)
18. `third_party/proton/CMakeLists.txt` - 1 fix (CUDA include directory)
19. `third_party/proton/Dialect/triton_proton.cc` - 1 fix (Windows guards)
20. `third_party/proton/csrc/include/Driver/Dispatch.h` - 1 fix (Windows LoadLibrary)
21. `third_party/proton/csrc/include/Driver/GPU/HsaApi.h` - 1 fix (__attribute__ workaround)
22. `third_party/proton/csrc/lib/Driver/GPU/HipApi.cpp` - 1 fix (Windows GetProcAddress)
23. `third_party/proton/csrc/lib/Driver/GPU/HsaApi.cpp` - 1 fix (Windows GetProcAddress)
24. `third_party/proton/csrc/lib/Driver/GPU/NvtxApi.cpp` - 1 fix (Windows _putenv_s)
25. `third_party/proton/csrc/lib/Driver/GPU/RoctracerApi.cpp` - 1 fix (Windows GetProcAddress)
26. `third_party/proton/csrc/lib/Profiler/Cupti/CuptiProfiler.cpp` - 1 fix (Windows _aligned_malloc)
27. `third_party/proton/csrc/lib/Profiler/RocTracer/RoctracerProfiler.cpp` - 1 fix (cxxabi.h conditional)

## Notes

- **LLVM build is a one-time setup** - once built for x64, you can reuse `C:\llvm-install`
- **All Triton fixes are REQUIRED** - Fixing LLVM architecture only solves linker errors. You still need all Windows compatibility fixes (Step 3) for Triton to compile on Windows.
- Environment variables must be set in the **same command prompt session** as the build
- Visual Studio environment (`vcvarsall.bat x64`) must be initialized before building
- The entire process (LLVM build + Triton install) can take **3-7 hours** total
- Ensure you have **50GB+ free disk space** for LLVM build
- **LLVM API Compatibility:** Triton is sensitive to LLVM versions because LLVM's API is not stable. If you build with a newer LLVM version (e.g., 22.0.0git) than what Triton was originally designed for, you'll need to apply the API compatibility fixes (Fix Category 3) in Step 3.

## Quick Reference: Environment Variables

**CUDA Variables:**
- `TRITON_PTXAS_PATH`
- `TRITON_PTXAS_BLACKWELL_PATH`
- `TRITON_CUOBJDUMP_PATH`
- `TRITON_NVDISASM_PATH`
- `TRITON_CUDACRT_PATH`
- `TRITON_CUDART_PATH`
- `TRITON_CUPTI_INCLUDE_PATH`
- `TRITON_CUPTI_LIB_PATH`

**LLVM Variables:**
- `LLVM_SYSPATH`
- `LLVM_INCLUDE_DIRS`
- `LLVM_LIBRARY_DIR`

**Build Variables:**
- `MAX_JOBS=1` (prevents OOM)
