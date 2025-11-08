# Triton & Flash Attention Windows Compatibility Fix

## Overview

This document provides a complete guide to fixing Triton and Flash Attention 2.0 on Windows. Triton uses Linux-specific commands and functions that fail on Windows, preventing Flash Attention 2.0 from working properly.

## Problem Summary

When running Flash Attention 2.0 with Triton on Windows, you may encounter several errors:

1. **FileNotFoundError**: `ldconfig` command not found (Linux-only)
2. **Compilation errors**: `-fPIC` flag not supported by MSVC
3. **Library linking errors**: `libcuda.so.1` not found (Linux library name)
4. **Header file errors**: `cuda.h` file not found during compilation
5. **Linker errors**: Error 1181/1104/1120 - cannot find `nvcuda.lib`, Python libraries, or unresolved symbols (Windows linking issues)
6. **Dynamic library errors**: `dlopen`/`dlsym` functions not available on Windows
7. **Memory allocation errors**: `posix_memalign` not available on Windows
8. **Grep command errors**: Unix `grep` command not available
9. **PTX assembler errors**: `ptxas.exe` or `ptxas-blackwell.exe` not found (CUDA tool path detection)
10. **File locking errors**: `PermissionError: [WinError 32]` when deleting temporary PTX files (Windows file locking)

## All Required Fixes

### Fix 1: NVIDIA Driver - CUDA Library Path Detection

**Problem:**
The `/sbin/ldconfig` command is Linux-only and doesn't exist on Windows, causing `FileNotFoundError` when Triton tries to find CUDA libraries.

**Solution:**
Replace `ldconfig` with Windows-compatible CUDA path detection that checks common CUDA installation directories.

**File to change:**
`<venv>\Lib\site-packages\triton\backends\nvidia\driver.py`

**Code (around line 25, in `libcuda_dirs()` function):**

Replace:
```python
libs = subprocess.check_output(["/sbin/ldconfig", "-p"]).decode(errors="ignore")
```

With:
```python
@functools.lru_cache()
def libcuda_dirs():
    if env_libcuda_path := knobs.nvidia.libcuda_path:
        return [env_libcuda_path]

    # Windows-compatible CUDA library detection
    import platform
    if platform.system() == "Windows":
        cuda_paths = []
        
        # Try common CUDA installation paths on Windows
        cuda_base_paths = [
            os.environ.get("CUDA_PATH", ""),
            os.environ.get("CUDA_HOME", ""),
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
        ]
        
        # Also check for versioned subdirectories
        for base_path in cuda_base_paths:
            if not base_path or not os.path.exists(base_path):
                continue
            
            # Check base path
            for subdir in ["bin", "lib", "lib64", "lib\\x64"]:
                full_path = os.path.join(base_path, subdir)
                if os.path.exists(full_path):
                    cuda_paths.append(full_path)
            
            # Check versioned subdirectories (e.g., v12.9, v12.8, etc.)
            try:
                for item in os.listdir(base_path):
                    versioned_path = os.path.join(base_path, item)
                    if os.path.isdir(versioned_path) and item.startswith("v"):
                        for subdir in ["bin", "lib", "lib64", "lib\\x64"]:
                            full_path = os.path.join(versioned_path, subdir)
                            if os.path.exists(full_path):
                                cuda_paths.append(full_path)
            except (OSError, PermissionError):
                pass
        
        # Remove duplicates while preserving order
        seen = set()
        unique_paths = []
        for path in cuda_paths:
            normalized = os.path.normpath(path)
            if normalized not in seen:
                seen.add(normalized)
                unique_paths.append(normalized)
        
        # On Windows, we don't check for libcuda.so.1 (that's a Linux library name)
        # Instead, we return the CUDA paths where nvcuda.dll would be found
        return unique_paths
    
    # Linux/Unix path: use ldconfig
    libs = subprocess.check_output(["/sbin/ldconfig", "-p"]).decode(errors="ignore")
    # each line looks like the following:
    # libcuda.so.1 (libc6,x86-64) => /lib/x86_64-linux-gnu/libcuda.so.1
    locs = [line.split()[-1] for line in libs.splitlines() if "libcuda.so.1" in line]
    dirs = [os.path.dirname(loc) for loc in locs]
    env_ld_library_path = os.getenv("LD_LIBRARY_PATH")
    if env_ld_library_path and not dirs:
        dirs = [dir for dir in env_ld_library_path.split(":") if os.path.exists(os.path.join(dir, "libcuda.so.1"))]
    msg = 'libcuda.so cannot found!\n'
    if locs:
        msg += 'Possible files are located at %s.' % str(locs)
        msg += 'Please create a symlink of libcuda.so to any of the files.'
    else:
        msg += 'Please make sure GPU is set up and then run "/sbin/ldconfig"'
        msg += ' (requires sudo) to refresh the linker cache.'
    assert any(os.path.exists(os.path.join(path, 'libcuda.so.1')) for path in dirs), msg
    return dirs
```

---

### Fix 2: NVIDIA Driver - CUDA Library Name

**Problem:**
Windows uses `nvcuda.dll`, not `libcuda.so.1`. The code tries to link against the Linux library name, causing linking errors.

**Solution:**
Use platform-specific library names: `nvcuda` on Windows, `libcuda.so.1` on Linux.

**File to change:**
`<venv>\Lib\site-packages\triton\backends\nvidia\driver.py`

**Code (around line 16, at top of file):**

Replace:
```python
libraries = ['libcuda.so.1']  # Linux library name
```

With:
```python
# Windows uses nvcuda.dll, Linux uses libcuda.so.1
import platform
if platform.system() == "Windows":
    libraries = ['nvcuda']
else:
    libraries = ['libcuda.so.1']
```

---

### Fix 3: NVIDIA Driver - CUDA Include Directories and Linking

**Problem:**
1. When compiling `cuda_utils.c`, the compiler cannot find `cuda.h` header file, causing `fatal error: 'cuda.h' file not found`.
2. CUDA driver API symbols are unresolved at link time, causing linker error 1120.

**Solution:**
1. Add a function to find CUDA include directories on Windows.
2. Link against `cuda.lib` (import library) on Windows to resolve CUDA driver API symbols.

**File to change:**
`<venv>\Lib\site-packages\triton\backends\nvidia\driver.py`

**Code:**

**1. Add function after `libcuda_dirs()` function (around line 110):**

```python
@functools.lru_cache()
def cuda_include_dirs():
    """Find CUDA include directories for compilation."""
    import platform
    if platform.system() == "Windows":
        cuda_include_paths = []
        
        # Try common CUDA installation paths on Windows
        cuda_base_paths = [
            os.environ.get("CUDA_PATH", ""),
            os.environ.get("CUDA_HOME", ""),
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
        ]
        
        # Check for versioned subdirectories and include directory
        for base_path in cuda_base_paths:
            if not base_path or not os.path.exists(base_path):
                continue
            
            # Check base path for include directory
            include_path = os.path.join(base_path, "include")
            if os.path.exists(include_path):
                cuda_include_paths.append(include_path)
            
            # Check versioned subdirectories (e.g., v12.9, v12.8, etc.)
            try:
                for item in os.listdir(base_path):
                    versioned_path = os.path.join(base_path, item)
                    if os.path.isdir(versioned_path) and item.startswith("v"):
                        include_path = os.path.join(versioned_path, "include")
                        if os.path.exists(include_path):
                            cuda_include_paths.append(include_path)
            except (OSError, PermissionError):
                pass
        
        # Remove duplicates while preserving order
        seen = set()
        unique_paths = []
        for path in cuda_include_paths:
            normalized = os.path.normpath(path)
            if normalized not in seen:
                seen.add(normalized)
                unique_paths.append(normalized)
        
        return unique_paths
    
    # Linux/Unix: try common CUDA paths
    linux_cuda_paths = [
        os.environ.get("CUDA_PATH", ""),
        os.environ.get("CUDA_HOME", ""),
        "/usr/local/cuda",
    ]
    
    include_paths = []
    for base_path in linux_cuda_paths:
        if not base_path or not os.path.exists(base_path):
            continue
        include_path = os.path.join(base_path, "include")
        if os.path.exists(include_path):
            include_paths.append(include_path)
    
    return include_paths
```

**2. Update `CudaUtils.__init__()` method (around line 179):**

Find:
```python
    def __init__(self):
        mod = compile_module_from_src(
            src=Path(os.path.join(dirname, "driver.c")).read_text(),
            name="cuda_utils",
            library_dirs=library_dirs(),
            include_dirs=include_dirs,
            libraries=libraries,
        )
```

Replace with:
```python
    def __init__(self):
        # On Windows, nvcuda.dll is loaded dynamically at runtime via LoadLibraryA
        # However, we still need to link against cuda.lib for the import library
        # to resolve CUDA driver API symbols at link time
        import platform
        if platform.system() == "Windows":
            # On Windows, link against cuda.lib (import library for nvcuda.dll)
            # This provides the symbols needed by the linker, even though the DLL is loaded dynamically
            compile_libraries = ["cuda"]
        else:
            compile_libraries = libraries
        mod = compile_module_from_src(
            src=Path(os.path.join(dirname, "driver.c")).read_text(),
            name="cuda_utils",
            library_dirs=library_dirs(),
            include_dirs=include_dirs + cuda_include_dirs(),
            libraries=compile_libraries,
        )
```

**3. Update launcher compilation (around line 829):**

Find:
```python
        mod = compile_module_from_src(
            src=src,
            name="__triton_launcher",
            library_dirs=library_dirs(),
            include_dirs=include_dirs,
            libraries=libraries,
        )
```

Replace with:
```python
        src = make_launcher(constants, signature, tensordesc_meta)
        # On Windows, nvcuda.dll is loaded dynamically at runtime via LoadLibraryA
        # However, we still need to link against cuda.lib for the import library
        # to resolve CUDA driver API symbols at link time
        import platform
        if platform.system() == "Windows":
            # On Windows, link against cuda.lib (import library for nvcuda.dll)
            compile_libraries = ["cuda"]
        else:
            compile_libraries = libraries
        mod = compile_module_from_src(
            src=src,
            name="__triton_launcher",
            library_dirs=library_dirs(),
            include_dirs=include_dirs + cuda_include_dirs(),
            libraries=compile_libraries,
        )
```

---

### Fix 4: Build System - Remove -fPIC Flag and Add Windows Support

**Problem:**
1. The `-fPIC` flag is not supported by MSVC on Windows, causing compilation errors.
2. Python libraries are not found during linking, causing error 1104.
3. Python symbols are unresolved, causing error 1120.
4. Architecture mismatch warnings (x64 vs x86).
5. Deprecated function warnings (`strcat`).
6. CUDA libraries may not link reliably using `-l` flags on Windows with clang.

**Solution:**
1. Remove `-fPIC` flag on Windows, add `-m64` for 64-bit compilation.
2. Add Python library directory detection and linking.
3. Suppress warnings on Windows.
4. Use full paths to `.lib` files on Windows when found (more reliable than `-l` flags with clang).

**File to change:**
`<venv>\Lib\site-packages\triton\runtime\build.py`

**Code (around line 43, in `_build()` function):**

Replace:
```python
cc_cmd = [cc, src, "-O3", "-shared", "-fPIC", "-Wno-psabi", "-o", so]
cc_cmd += [_library_flag(lib) for lib in libraries]
cc_cmd += [f"-L{dir}" for dir in library_dirs]
cc_cmd += [f"-I{dir}" for dir in include_dirs if dir is not None]
cc_cmd.extend(ccflags)
```

With:
```python
    import platform
    cc_cmd = [cc, src, "-O3", "-shared"]
    # -fPIC is not supported on Windows MSVC
    if platform.system() != "Windows":
        cc_cmd.append("-fPIC")
    else:
        # On Windows, ensure we compile for 64-bit (x64) architecture
        # This is required because Python libraries are x64
        cc_cmd.append("-m64")
    cc_cmd.append("-Wno-psabi")
    # On Windows, suppress deprecated function warnings and add necessary flags
    if platform.system() == "Windows":
        cc_cmd.append("-Wno-deprecated-declarations")
        # Define _CRT_SECURE_NO_WARNINGS to suppress strcat warnings
        cc_cmd.append("-D_CRT_SECURE_NO_WARNINGS")
    cc_cmd.append("-o")
    cc_cmd.append(so)
    # On Windows, try to use full paths for .lib files for more reliable linking
    if platform.system() == "Windows":
        windows_libs = []
        remaining_libs = []
        for lib in libraries:
            # Check if it's a .lib file or a library name that should be found as .lib
            lib_file = None
            if lib.endswith(".lib"):
                # Already a full path or filename
                if os.path.exists(lib) or os.path.isabs(lib):
                    windows_libs.append(lib)
                    continue
                # Try to find it in library directories
                for lib_dir in library_dirs:
                    potential_path = os.path.join(lib_dir, lib)
                    if os.path.exists(potential_path):
                        lib_file = potential_path
                        break
            else:
                # Library name like "cuda" - try to find cuda.lib in library directories
                for lib_dir in library_dirs:
                    potential_path = os.path.join(lib_dir, f"{lib}.lib")
                    if os.path.exists(potential_path):
                        lib_file = potential_path
                        break
            
            if lib_file:
                windows_libs.append(lib_file)
            else:
                # Fall back to regular -l flag if not found
                remaining_libs.append(lib)
        
        # Add full paths for found libraries, and -l flags for others
        cc_cmd.extend(windows_libs)
        cc_cmd += [_library_flag(lib) for lib in remaining_libs]
    else:
        cc_cmd += [_library_flag(lib) for lib in libraries]
    cc_cmd += [f"-L{dir}" for dir in library_dirs]
    # On Windows, add Python library directory for linking
    if platform.system() == "Windows":
        import sys
        import sysconfig
        py_lib_dir = sysconfig.get_config_var("LIBDIR")
        if not py_lib_dir:
            # Try to find Python libs directory
            # Check include directory path first (most reliable for pyenv, system Python, etc.)
            # The include directory is in the base Python installation
            scheme = sysconfig.get_default_scheme()
            include_dir = sysconfig.get_paths(scheme=scheme).get("include")
            if include_dir:
                base_from_include = os.path.dirname(include_dir)
                libs_from_include = os.path.join(base_from_include, "libs")
                if os.path.exists(libs_from_include):
                    py_lib_dir = libs_from_include
            
            if not py_lib_dir:
                # Fallback: Check base Python installation (for pyenv, system Python, etc.)
                py_base = sysconfig.get_config_var("base")
                py_prefix = sysconfig.get_config_var("prefix") or sysconfig.get_paths(scheme=scheme).get("data", "")
                # Get Python executable directory (for pyenv installations)
                py_exec_dir = os.path.dirname(os.path.dirname(sys.executable)) if hasattr(sys, 'executable') else None
                
                # Common locations for Python libs on Windows
                possible_lib_dirs = []
                if py_base:
                    possible_lib_dirs.append(os.path.join(py_base, "libs"))
                if py_exec_dir:
                    possible_lib_dirs.append(os.path.join(py_exec_dir, "libs"))
                if py_prefix:
                    possible_lib_dirs.extend([
                        os.path.join(py_prefix, "libs"),
                        os.path.join(os.path.dirname(py_prefix), "libs"),
                    ])
                
                for lib_dir in possible_lib_dirs:
                    if lib_dir and os.path.exists(lib_dir):
                        py_lib_dir = lib_dir
                        break
        if py_lib_dir and os.path.exists(py_lib_dir):
            cc_cmd.append(f"-L{py_lib_dir}")
            # On Windows, explicitly link against Python library
            # Try python312.lib, python3.lib, or python.lib
            # On Windows with clang, use full path to .lib file for more reliable linking
            import sys
            py_version = f"{sys.version_info.major}{sys.version_info.minor}"
            py_lib_names = [f"python{py_version}", f"python{sys.version_info.major}", "python3", "python"]
            for lib_name in py_lib_names:
                lib_file = os.path.join(py_lib_dir, f"{lib_name}.lib")
                if os.path.exists(lib_file):
                    # Use full path on Windows for more reliable linking
                    cc_cmd.append(lib_file)
                    break
    cc_cmd += [f"-I{dir}" for dir in include_dirs if dir is not None]
    cc_cmd.extend(ccflags)
    # Capture both stdout and stderr to see linker errors if compilation fails
    try:
        result = subprocess.run(cc_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if result.returncode != 0:
            # Compilation failed, print the error output
            error_output = result.stdout if result.stdout else "No error output"
            if error_output and len(error_output.strip()) > 0:
                import logging
                log = logging.getLogger(__name__)
                log.error(f"Compilation failed. Linker errors:\n{error_output}")
            raise subprocess.CalledProcessError(result.returncode, cc_cmd, output=result.stdout)
    except subprocess.CalledProcessError as e:
        raise
```

---

### Fix 5: NVIDIA Driver C Code - Windows Compatibility

**Problem:**
The C code uses Unix-only functions:
- `dlopen`/`dlsym`/`dlclose` (dynamic library loading)
- `posix_memalign` (aligned memory allocation)

**Solution:**
Use Windows equivalents: `LoadLibrary`/`GetProcAddress` for dynamic loading, `_aligned_malloc`/`_aligned_free` for aligned memory.

**File to change:**
`<venv>\Lib\site-packages\triton\backends\nvidia\driver.c`

**Code:**

**1. Conditional includes (lines 1-6):**

Replace:
```c
#include "cuda.h"
#include <dlfcn.h>
```

With:
```c
#include "cuda.h"
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif
```

**2. Windows dynamic library loading (around line 181):**

**CRITICAL**: You cannot use `#ifdef` inside a macro definition. Define two separate macros using conditional compilation.

Find:
```c
#define defineGetFunctionHandle(name, symbolName)                              \
  static symbolName##_t name() {                                               \
    /* Open the shared library */                                              \
#ifdef _WIN32
    HMODULE libHandle = LoadLibraryA("nvcuda.dll");                            \
    ...
#else
    void *libHandle = dlopen("libcuda.so.1", RTLD_LAZY);                       \
    ...
#endif
    return funcHandle;                                                         \
  }
```

Replace with:
```c
#ifdef _WIN32
#define defineGetFunctionHandle(name, symbolName)                              \
  static symbolName##_t name() {                                               \
    /* Open the shared library */                                              \
    HMODULE libHandle = LoadLibraryA("nvcuda.dll");                            \
    if (!libHandle) {                                                           \
      DWORD error = GetLastError();                                              \
      char errMsg[256];                                                          \
      snprintf(errMsg, sizeof(errMsg), "Failed to open nvcuda.dll. Error: %lu", error); \
      PyErr_SetString(PyExc_RuntimeError, errMsg);                              \
      return NULL;                                                              \
    }                                                                           \
    symbolName##_t funcHandle = (symbolName##_t)GetProcAddress(libHandle, #symbolName); \
    if (!funcHandle) {                                                          \
      DWORD error = GetLastError();                                              \
      char errMsg[256];                                                         \
      snprintf(errMsg, sizeof(errMsg), "Failed to retrieve " #symbolName " from nvcuda.dll. Error: %lu", error); \
      PyErr_SetString(PyExc_RuntimeError, errMsg);                              \
      FreeLibrary(libHandle);                                                   \
      return NULL;                                                              \
    }                                                                           \
    return funcHandle;                                                         \
  }
#else
#define defineGetFunctionHandle(name, symbolName)                              \
  static symbolName##_t name() {                                               \
    /* Open the shared library */                                              \
    void *libHandle = dlopen("libcuda.so.1", RTLD_LAZY);                       \
    if (!libHandle) {                                                          \
      PyErr_SetString(PyExc_RuntimeError, "Failed to open libcuda.so.1");      \
      return NULL;                                                             \
    }                                                                          \
    /* Clear any existing error */                                             \
    dlerror();                                                                 \
    symbolName##_t funcHandle = (symbolName##_t)dlsym(libHandle, #symbolName); \
    /* Check for errors */                                                     \
    const char *err = dlerror();                                               \
    if (err) {                                                                 \
      PyErr_SetString(PyExc_RuntimeError,                                      \
                      "Failed to retrieve " #symbolName " from libcuda.so.1"); \
      dlclose(libHandle);                                                      \
      return NULL;                                                             \
    }                                                                          \
    return funcHandle;                                                         \
  }
#endif
```

**3. Aligned memory allocation (around line 298):**

Find:
```c
    void* ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to allocate aligned memory");
        return NULL;
    }
```

Replace with:
```c
#ifdef _WIN32
    void* ptr = _aligned_malloc(size, alignment);
#else
    void* ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to allocate aligned memory");
        return NULL;
    }
#endif
```

**4. Aligned memory deallocation (around line 312):**

Find:
```c
    free(ptr);
```

Replace with:
```c
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
```

---

### Fix 6: Build Extern Tool - Grep Command

**Problem:**
The `grep` command is Unix-only and not available on Windows, causing `FileNotFoundError`.

**Solution:**
Use Python file reading on Windows instead of the `grep` command.

**File to change:**
`<venv>\Lib\site-packages\triton\tools\build_extern.py`

**Code (around line 255, in `parse_symbols()` method):**

Add `import platform` at the top of the file if not already present.

Find:
```python
    def parse_symbols(self, input_file) -> None:
        if len(self.symbols) > 0:
            return
        output = subprocess.check_output(["grep", "define", input_file]).decode().splitlines()
```

Replace with:
```python
    def parse_symbols(self, input_file) -> None:
        if len(self.symbols) > 0:
            return
        # Windows-compatible: use Python file reading instead of grep
        if platform.system() == "Windows":
            with open(input_file, 'r', encoding='utf-8') as f:
                output = [line for line in f if "define" in line]
        else:
            output = subprocess.check_output(["grep", "define", input_file]).decode().splitlines()
```

---

### Fix 7: AMD Driver - HIP Include Directories

**Problem:**
When compiling `hip_utils.c`, the compiler may not find HIP header files (`hip/hip_runtime.h`, `hip/hip_runtime_api.h`) if HIP/ROCm include directories are not in the compiler's include paths.

**Solution:**
Add a function to find HIP/ROCm include directories on Windows (similar to Fix 3 for CUDA).

**File to change:**
`<venv>\Lib\site-packages\triton\backends\amd\driver.py`

**Code:**

**1. Add function after `include_dirs` definition (around line 200):**

```python
@functools.lru_cache()
def hip_include_dirs():
    """Find HIP/ROCm include directories for compilation."""
    import platform
    if platform.system() == "Windows":
        hip_include_paths = []
        
        # Try common ROCm installation paths on Windows
        rocm_base_paths = [
            os.environ.get("ROCM_PATH", ""),
            r"C:\Program Files\AMD\ROCm",
            r"C:\ROCm",
        ]
        
        # Check for versioned subdirectories and include directory
        for base_path in rocm_base_paths:
            if not base_path or not os.path.exists(base_path):
                continue
            
            # Check base path for include directory
            include_path = os.path.join(base_path, "include")
            if os.path.exists(include_path):
                hip_include_paths.append(include_path)
            
            # Check versioned subdirectories (e.g., 6.0, 5.7, etc.)
            try:
                for item in os.listdir(base_path):
                    versioned_path = os.path.join(base_path, item)
                    if os.path.isdir(versioned_path):
                        include_path = os.path.join(versioned_path, "include")
                        if os.path.exists(include_path):
                            hip_include_paths.append(include_path)
            except (OSError, PermissionError):
                pass
        
        # Remove duplicates while preserving order
        seen = set()
        unique_paths = []
        for path in hip_include_paths:
            normalized = os.path.normpath(path)
            if normalized not in seen:
                seen.add(normalized)
                unique_paths.append(normalized)
        
        return unique_paths
    
    # Linux/Unix: try common ROCm paths
    linux_rocm_paths = [
        os.environ.get("ROCM_PATH", ""),
        "/opt/rocm",
        "/usr/local/rocm",
    ]
    
    include_paths = []
    for base_path in linux_rocm_paths:
        if not base_path or not os.path.exists(base_path):
            continue
        include_path = os.path.join(base_path, "include")
        if os.path.exists(include_path):
            include_paths.append(include_path)
    
    return include_paths
```

**2. Update `HIPUtils.__init__()` method (around line 209):**

Find:
```python
            include_dirs=include_dirs,
```

Replace with:
```python
            include_dirs=include_dirs + hip_include_dirs(),
```

**3. Update launcher compilation (around line 750):**

Find:
```python
            include_dirs=include_dirs,
```

Replace with:
```python
            include_dirs=include_dirs + hip_include_dirs(),
```

---

### Fix 8: PTX Assembler - Use Regular ptxas for All Architectures and CUDA Path Detection

**Problem:**
1. For architectures >= 100 (Blackwell), Triton tries to use `ptxas-blackwell.exe`, which may not exist, causing `RuntimeError: Cannot find ptxas-blackwell.exe`.
2. Triton looks for `ptxas.exe` in its own `backends/nvidia/bin` directory, but on Windows it's in the CUDA installation directory, causing `RuntimeError: Cannot find ptxas.exe`.

**Solution:**
1. Use regular `ptxas` for all architectures (CUDA 12.9+ supports all architectures including Blackwell).
2. Add CUDA path detection to find `ptxas.exe` in CUDA installation.

**File to change:**
`<venv>\Lib\site-packages\triton\backends\nvidia\compiler.py`

**Code (around line 34, in `get_ptxas()` function):**

Find:
```python
def get_ptxas(arch: int) -> knobs.NvidiaTool:
    return knobs.nvidia.ptxas_blackwell if arch >= 100 else knobs.nvidia.ptxas
```

Replace with:
```python
def get_ptxas(arch: int) -> knobs.NvidiaTool:
    # Use regular ptxas for all architectures
    # Regular ptxas in CUDA 12.9+ supports all architectures including Blackwell (compute capability 12.0)
    # ptxas-blackwell is only needed for very specific cases and may not be available
    return knobs.nvidia.ptxas
```

**File to change:**
`<venv>\Lib\site-packages\triton\knobs.py`

**Code (around line 204, in `env_nvidia_tool.transform()` method):**

Add `import platform` at the top of the file if not already present.

Find:
```python
    def transform(self, path: str) -> NvidiaTool:
        # We still add default as fallback in case the pointed binary isn't
        # accessible.
        if path is not None:
            paths = [path, self.default_path]
        else:
            paths = [self.default_path]

        for path in paths:
            if tool := NvidiaTool.from_path(path):
                return tool

        raise RuntimeError(f"Cannot find {self.binary}")
```

Replace with:
```python
    def transform(self, path: str) -> NvidiaTool:
        # We still add default as fallback in case the pointed binary isn't
        # accessible.
        if path is not None:
            paths = [path, self.default_path]
        else:
            paths = [self.default_path]
        
        # On Windows, also check CUDA installation paths if default path doesn't exist
        if platform.system() == "Windows" and not os.path.exists(self.default_path):
            # Try to find the tool in CUDA installation
            cuda_base_paths = [
                os.environ.get("CUDA_PATH", ""),
                os.environ.get("CUDA_HOME", ""),
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
            ]
            for base_path in cuda_base_paths:
                if not base_path or not os.path.exists(base_path):
                    continue
                # Check base path
                cuda_bin_path = os.path.join(base_path, "bin", self.binary)
                if os.path.exists(cuda_bin_path):
                    paths.insert(0, cuda_bin_path)
                # Check versioned subdirectories
                try:
                    for item in os.listdir(base_path):
                        versioned_path = os.path.join(base_path, item)
                        if os.path.isdir(versioned_path) and item.startswith("v"):
                            cuda_bin_path = os.path.join(versioned_path, "bin", self.binary)
                            if os.path.exists(cuda_bin_path):
                                paths.insert(0, cuda_bin_path)
                except (OSError, PermissionError):
                    pass

        for path in paths:
            if tool := NvidiaTool.from_path(path):
                return tool

        raise RuntimeError(f"Cannot find {self.binary}")
```

---

### Fix 9: Windows File Locking - Skip Temporary PTX File Deletion

**Problem:**
On Windows, after `ptxas.exe` completes, the temporary PTX file may still be locked by the process, causing `PermissionError: [WinError 32] The process cannot access the file because it is being used by another process` when trying to delete it.

**Solution:**
Skip file deletion on Windows entirely. The temporary files are in the system temp directory (`%TEMP%`) and Windows will automatically clean them up, so immediate deletion is not necessary.

**File to change:**
`<venv>\Lib\site-packages\triton\backends\nvidia\compiler.py`

**Code:**

**1. Add import at the top of the file (around line 16):**

Add:
```python
import platform
```

**2. In `make_cubin()` method (around line 495):**

Find:
```python
            try:
                subprocess.run(ptxas_cmd, check=True, close_fds=False, stderr=flog)
                if knobs.nvidia.dump_ptxas_log:
                    with open(flog.name) as log_file:
                        print(log_file.read())

                if os.path.exists(fsrc.name):
                    os.remove(fsrc.name)
                if os.path.exists(flog.name):
                    os.remove(flog.name)
            except subprocess.CalledProcessError as e:
                with open(flog.name) as log_file:
                    log = log_file.read()
                if os.path.exists(flog.name):
                    os.remove(flog.name)
```

Replace with:
```python
            try:
                subprocess.run(ptxas_cmd, check=True, close_fds=False, stderr=flog)
                if knobs.nvidia.dump_ptxas_log:
                    with open(flog.name) as log_file:
                        print(log_file.read())

                # On Windows, skip deletion - files may be locked and OS will clean up temp files anyway
                # On other platforms, delete to keep temp directory clean
                if platform.system() != "Windows":
                    if os.path.exists(fsrc.name):
                        os.remove(fsrc.name)
                    if os.path.exists(flog.name):
                        os.remove(flog.name)
            except subprocess.CalledProcessError as e:
                with open(flog.name) as log_file:
                    log = log_file.read()
                # On Windows, skip deletion - files may be locked and OS will clean up temp files anyway
                if platform.system() != "Windows":
                    if os.path.exists(flog.name):
                        os.remove(flog.name)
```

---

## Files Modified Summary

### NVIDIA Backend (Required for NVIDIA GPUs)
1. **`backends/nvidia/driver.py`** - Fixes 1, 2, 3
2. **`backends/nvidia/driver.c`** - Fix 5
3. **`backends/nvidia/compiler.py`** - Fixes 8, 9

### Build System (Required)
4. **`runtime/build.py`** - Fix 4

### Tool Detection (Required)
5. **`knobs.py`** - Fix 8

### Optional Fixes
6. **`tools/build_extern.py`** - Fix 6 (only if using build_extern tool)

### AMD Backend (For AMD GPUs)
7. **`backends/amd/driver.py`** - Fix 7
8. **`backends/amd/driver.c`** - Already has Windows compatibility (conditional includes and LoadLibrary)

**Note**: AMD driver files in newer Triton versions may already include Windows compatibility for:
- ✅ Windows detection for `ldconfig` calls (already fixed)
- ✅ Conditional includes (`#ifdef _WIN32`) in generated C code (already fixed)
- ✅ Windows-compatible library loading (`LoadLibrary` vs `dlopen`) (already fixed)
- ⚠️ HIP include directories (Fix 7 - now added)

## Verification

After applying all fixes:

1. **Clear Python cache:**
   ```powershell
   Remove-Item -Recurse -Force "$env:VIRTUAL_ENV\Lib\site-packages\triton\**\__pycache__"
   ```

2. **Test your script:**
   ```python
   from transformers import Qwen2_5OmniForConditionalGeneration
   model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
       "Qwen/Qwen2.5-Omni-3B",
       device_map="auto",
       torch_dtype=torch.bfloat16,
       attn_implementation="flash_attention_2",
   )
   ```

3. **Expected result:** No `FileNotFoundError`, no compilation errors, Flash Attention 2.0 works correctly.

## Important Notes

- **Reapply after reinstalling Triton**: If you reinstall or update Triton, you'll need to apply these fixes again
- **Backward compatible**: All fixes preserve Linux/Unix behavior for non-Windows systems
- **CUDA required**: Ensure CUDA Toolkit is installed and `CUDA_PATH` environment variable is set (optional but recommended)
- **Fix 6 is optional**: Only needed if you use the `build_extern.py` tool

## Troubleshooting

### CUDA not found
- Verify CUDA Toolkit is installed
- Check `CUDA_PATH` environment variable
- Ensure CUDA installation path exists: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\include`

### Linker errors (1104, 1120, 1181)
- **Error 1181**: Usually means `nvcuda.lib` not found - fixed by linking against `cuda.lib` on Windows
- **Error 1104**: Usually means Python library directory not found - fixed by adding Python library directory to linker search path
  - The fix now checks multiple locations including base Python installation (important for pyenv, system Python)
  - For pyenv installations, libraries are typically at: `C:\Users\<user>\.pyenv\pyenv-win\versions\<version>\libs`
  - Verify Python installation has a `libs` directory containing `python<version>.lib`
- **Error 1120**: Usually means unresolved symbols or architecture mismatch - fixed by:
  - Adding `-m64` flag to ensure 64-bit compilation (fixes "library machine type 'x64' conflicts with target machine type 'x86'" warning)
  - Explicitly linking against Python library (`python312.lib`, `python3.lib`, or `python.lib`)
  - Linking against `cuda.lib` import library (resolves CUDA driver API symbols)
  - Uses full path to library file on Windows for more reliable linking with clang (applies to both Python and CUDA libraries)
  - Automatically detects and uses full paths to `.lib` files when found in library directories
  - Error capture added to show actual linker errors if compilation still fails
- **PTX assembler errors**: `RuntimeError: Cannot find ptxas.exe` or `ptxas-blackwell.exe` - fixed by:
  - Using regular `ptxas` for all architectures (CUDA 12.9+ supports all architectures including Blackwell)
  - Automatically detecting CUDA installation paths to find `ptxas.exe`
  - No need to download or install `ptxas-blackwell` separately
- Check that CUDA is properly installed

### Still getting errors
- Clear Python cache (see Verification section)
- Verify all fixes were applied correctly
- Check that you're using a compatible Python version (3.8+)
- Ensure you have a C compiler installed (Visual Studio Build Tools or Clang)
- Try running with verbose output: Set `TRITON_VERBOSE=1` environment variable

## Additional Notes

### AMD GPU Support

If you're using AMD GPUs (ROCm), the AMD driver files may already have Windows compatibility in newer Triton versions. However, if you encounter issues:

1. **Check `backends/amd/driver.py`**:
   - Should have Windows detection before `ldconfig` calls (similar to Fix 1)
   - Should skip `libc.so.6` loading on Windows (similar to Fix 2)

2. **Check `backends/amd/driver.c`**:
   - Should have conditional includes (`#ifdef _WIN32`)
   - Should use `LoadLibrary`/`GetProcAddress` on Windows instead of `dlopen`/`dlsym`

3. **Check generated C code in `backends/amd/driver.py`**:
   - Should have conditional includes in the template
   - Should use Windows APIs in the `#ifdef _WIN32` block

### Verification Checklist

After applying fixes, verify:
- [ ] No `FileNotFoundError` for `ldconfig`
- [ ] No compilation errors for `-fPIC` flag
- [ ] No errors about `libcuda.so.1` or `nvcuda.dll` not found
- [ ] No errors about `cuda.h` not found
- [ ] No errors about `dlopen`/`dlsym` functions
- [ ] No errors about `posix_memalign`
- [ ] No linker errors (1104, 1120, 1181)
- [ ] No PTX assembler errors (ptxas.exe or ptxas-blackwell.exe not found)
- [ ] No file locking errors (PermissionError when deleting temp files)
- [ ] No deprecated function warnings (strcat)
- [ ] Flash Attention 2.0 loads successfully
- [ ] Model inference works with Flash Attention 2.0

## Status

✅ **All fixes documented** - Use the automated script (`fix_triton_windows.ps1`) to apply all fixes at once.

**Script Coverage:**
- ✅ Fix 1: NVIDIA driver - CUDA library path detection
- ✅ Fix 2: NVIDIA driver - Library name
- ✅ Fix 3: NVIDIA driver - CUDA include directories + Link against cuda.lib on Windows
- ✅ Fix 4: Build system - Remove -fPIC flag, add -m64, suppress warnings, add Python library paths
- ✅ Fix 5: NVIDIA driver.c - Windows compatibility (macro fix)
- ✅ Fix 6: Build extern tool - Grep command
- ✅ Fix 7: AMD driver - HIP include directories
- ✅ Fix 8: PTX assembler - Use regular ptxas for all architectures + CUDA path detection
- ✅ Fix 9: Windows file locking - Skip temporary PTX file deletion on Windows

**Note**: The script now handles both NVIDIA and AMD backend fixes. AMD driver.c already has Windows compatibility, but the script adds HIP include directory detection.

**Additional Fixes Applied:**
- **Link against cuda.lib on Windows**: Prevents linker error 1120 (unresolved CUDA driver API symbols)
- **Add -m64 flag**: Prevents architecture mismatch (x64 vs x86) errors
- **Python library directory**: Prevents linker error 1104 (Python libraries not found)
- **Explicit Python library linking**: Prevents linker error 1120 (unresolved Python symbols)
- **Full paths for Windows libraries**: Uses full paths to `.lib` files when found (more reliable than `-l` flags with clang on Windows)
  - Automatically detects `cuda.lib` and other `.lib` files in library directories
  - Falls back to `-l` flags if library file not found
  - Improves linking reliability for both CUDA and other libraries on Windows
- **Warning suppression**: Suppresses deprecated `strcat` warnings on Windows
- **PTX assembler path detection**: Automatically finds ptxas.exe in CUDA installation
- **PTX assembler selection**: Uses regular ptxas for all architectures (no need for ptxas-blackwell)
- **Skip file deletion on Windows**: Prevents PermissionError when deleting temporary PTX files (Windows cleans up temp files automatically)
