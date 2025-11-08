# Triton & Flash Attention Windows Compatibility Fix

## Overview

This document provides a complete guide to fixing Triton and Flash Attention 2.0 on Windows. Triton uses Linux-specific commands and functions that fail on Windows, preventing Flash Attention 2.0 from working properly.

## Problem Summary

When running Flash Attention 2.0 with Triton on Windows, you may encounter several errors:

1. **FileNotFoundError**: `ldconfig` command not found (Linux-only)
2. **Compilation errors**: `-fPIC` flag not supported by MSVC
3. **Library linking errors**: `libcuda.so.1` not found (Linux library name)
4. **Header file errors**: `cuda.h` file not found during compilation
5. **Linker errors**: Error 1181/1104 - cannot find `nvcuda.lib` or Python libraries (Windows linking issues)
6. **Dynamic library errors**: `dlopen`/`dlsym` functions not available on Windows
7. **Memory allocation errors**: `posix_memalign` not available on Windows
8. **Grep command errors**: Unix `grep` command not available

## All Required Fixes

### Fix 1: NVIDIA Driver - CUDA Library Path Detection

**File:** `<venv>\Lib\site-packages\triton\backends\nvidia\driver.py`  
**Location:** `libcuda_dirs()` function (around line 25)

**Problem:**
```python
libs = subprocess.check_output(["/sbin/ldconfig", "-p"]).decode(errors="ignore")
```
The `/sbin/ldconfig` command is Linux-only and doesn't exist on Windows, causing `FileNotFoundError`.

**Solution:**
Replace with Windows-compatible CUDA path detection:

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

**File:** `<venv>\Lib\site-packages\triton\backends\nvidia\driver.py`  
**Location:** Top of file (around line 16)

**Problem:**
```python
libraries = ['libcuda.so.1']  # Linux library name
```
Windows uses `nvcuda.dll`, not `libcuda.so.1`.

**Solution:**
```python
# Windows uses nvcuda.dll, Linux uses libcuda.so.1
import platform
if platform.system() == "Windows":
    libraries = ['nvcuda']
else:
    libraries = ['libcuda.so.1']
```

---

### Fix 3: NVIDIA Driver - CUDA Include Directories

**File:** `<venv>\Lib\site-packages\triton\backends\nvidia\driver.py`  
**Location:** After `libcuda_dirs()` function

**Problem:**
When compiling `cuda_utils.c`, the compiler cannot find `cuda.h`:
```
fatal error: 'cuda.h' file not found
```

**Solution:**
Add function to find CUDA include directories:

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

Then update `compile_module_from_src` calls to include CUDA include directories:

**Find (around line 184):**
```python
            include_dirs=include_dirs,
```

**Replace with:**
```python
            include_dirs=include_dirs + cuda_include_dirs(),
```

Apply this change in two places:
1. `CudaUtils.__init__()` method (around line 184)
2. Launcher compilation (around line 830)

**Important**: On Windows, also skip linking against `nvcuda` at compile time since it's loaded dynamically at runtime. This prevents linker errors (error 1181) where `nvcuda.lib` may not be available.

**In `CudaUtils.__init__()` method (around line 179), update:**
```python
    def __init__(self):
        # On Windows, nvcuda.dll is loaded dynamically at runtime via LoadLibraryA
        # so we don't need to link against it at compile time
        import platform
        compile_libraries = libraries if platform.system() != "Windows" else []
        mod = compile_module_from_src(
            src=Path(os.path.join(dirname, "driver.c")).read_text(),
            name="cuda_utils",
            library_dirs=library_dirs(),
            include_dirs=include_dirs + cuda_include_dirs(),
            libraries=compile_libraries,
        )
```

**In launcher compilation (around line 829), also update:**
```python
        src = make_launcher(constants, signature, tensordesc_meta)
        # On Windows, nvcuda.dll is loaded dynamically at runtime via LoadLibraryA
        # so we don't need to link against it at compile time
        import platform
        compile_libraries = libraries if platform.system() != "Windows" else []
        mod = compile_module_from_src(
            src=src,
            name="__triton_launcher",
            library_dirs=library_dirs(),
            include_dirs=include_dirs + cuda_include_dirs(),
            libraries=compile_libraries,
        )
```

This prevents linker errors (error 1181) on Windows where `nvcuda.lib` may not be available.

---

### Fix 4: Build System - Remove -fPIC Flag and Add Windows Support

**File:** `<venv>\Lib\site-packages\triton\runtime\build.py`  
**Location:** `_build()` function (around line 43)

**Problem:**
```python
cc_cmd = [cc, src, "-O3", "-shared", "-fPIC", "-Wno-psabi", "-o", so]
```
The `-fPIC` flag is not supported by MSVC on Windows, causing compilation errors. Additionally, Windows may need Python library directories and warning suppressions.

**Solution:**
```python
    import platform
    cc_cmd = [cc, src, "-O3", "-shared"]
    # -fPIC is not supported on Windows MSVC
    if platform.system() != "Windows":
        cc_cmd.append("-fPIC")
    cc_cmd.append("-Wno-psabi")
    # On Windows, suppress deprecated function warnings and add necessary flags
    if platform.system() == "Windows":
        cc_cmd.append("-Wno-deprecated-declarations")
        # Define _CRT_SECURE_NO_WARNINGS to suppress strcat warnings
        cc_cmd.append("-D_CRT_SECURE_NO_WARNINGS")
    cc_cmd.append("-o")
    cc_cmd.append(so)
    cc_cmd += [_library_flag(lib) for lib in libraries]
    cc_cmd += [f"-L{dir}" for dir in library_dirs]
    # On Windows, add Python library directory for linking
    if platform.system() == "Windows":
        import sys
        py_lib_dir = sysconfig.get_config_var("LIBDIR")
        if not py_lib_dir:
            # Try to find Python libs directory
            # Check include directory path first (most reliable for pyenv, system Python, etc.)
            # The include directory is in the base Python installation
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
            import sys
            py_version = f"{sys.version_info.major}{sys.version_info.minor}"
            py_lib_names = [f"python{py_version}", f"python{sys.version_info.major}", "python3", "python"]
            for lib_name in py_lib_names:
                lib_file = os.path.join(py_lib_dir, f"{lib_name}.lib")
                if os.path.exists(lib_file):
                    cc_cmd.append(f"-l{lib_name}")
                    break
    cc_cmd += [f"-I{dir}" for dir in include_dirs if dir is not None]
    cc_cmd.extend(ccflags)
```

This fix:
1. Removes `-fPIC` flag on Windows (not supported by MSVC)
2. Suppresses deprecated function warnings (`-Wno-deprecated-declarations`)
3. Defines `_CRT_SECURE_NO_WARNINGS` to suppress `strcat` warnings
4. Adds Python library directory to linker search path (prevents error 1104)
   - Checks `sysconfig.get_config_var("LIBDIR")` first
   - **Most reliable**: Derives libs directory from include directory path (works for pyenv, system Python, venv)
   - Fallback: Checks base Python installation directory
   - Fallback: Checks Python executable directory
   - Fallback: Checks prefix paths (venv and other locations)
5. Explicitly links against Python library (prevents error 1120)
   - Automatically detects Python version and links against `python312.lib`, `python3.lib`, or `python.lib`
   - Tries version-specific library first (e.g., `python312`), then generic names

---

### Fix 5: NVIDIA Driver C Code - Windows Compatibility

**File:** `<venv>\Lib\site-packages\triton\backends\nvidia\driver.c`  
**Location:** Multiple locations

**Problem:**
The C code uses Unix-only functions:
- `dlopen`/`dlsym`/`dlclose` (dynamic library loading)
- `posix_memalign` (aligned memory allocation)

**Solution:**

**1. Conditional includes (lines 1-6):**
```c
#include "cuda.h"
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif
```

**2. Windows dynamic library loading (around line 181):**
Replace `defineGetFunctionHandle` macro. **CRITICAL**: You cannot use `#ifdef` inside a macro definition. Instead, define two separate macros using conditional compilation:

**Find:**
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

**Replace with:**
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

**Important**: The `#ifdef _WIN32` must be OUTSIDE the macro definition, not inside it. This is a common mistake that causes compilation errors.

**3. Aligned memory allocation (around line 298):**
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
```c
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
```

---

### Fix 6: Build Extern Tool - Grep Command

**File:** `<venv>\Lib\site-packages\triton\tools\build_extern.py`  
**Location:** `parse_symbols()` method (around line 255)

**Problem:**
```python
output = subprocess.check_output(["grep", "define", input_file]).decode().splitlines()
```
The `grep` command is Unix-only and not available on Windows.

**Solution:**
Add `import platform` at the top, then:

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
        for line in output:
            symbol = self._extract_symbol(line)
            if symbol is None:
                continue
            self._symbols[symbol.name] = symbol

        self._group_symbols()
```

---

### Fix 7: AMD Driver - HIP Include Directories

**File:** `<venv>\Lib\site-packages\triton\backends\amd\driver.py`  
**Location:** After `include_dirs` definition

**Problem:**
When compiling `hip_utils.c`, the compiler may not find HIP header files (`hip/hip_runtime.h`, `hip/hip_runtime_api.h`) if HIP/ROCm include directories are not in the compiler's include paths.

**Solution:**
Add function to find HIP/ROCm include directories (similar to Fix 3 for CUDA):

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

Then update `compile_module_from_src` calls to include HIP include directories:

**Find (around line 209 and 750):**
```python
            include_dirs=include_dirs,
```

**Replace with:**
```python
            include_dirs=include_dirs + hip_include_dirs(),
```

Apply this change in two places:
1. `HIPUtils.__init__()` method (around line 209)
2. Launcher compilation (around line 750)

**Note**: AMD driver files may already have Windows compatibility for `ldconfig` and `libc.so.6` in newer Triton versions. This fix adds HIP include directory detection.

---

## Files Modified Summary

### NVIDIA Backend (Required for NVIDIA GPUs)
1. **`backends/nvidia/driver.py`** - Fixes 1, 2, 3
2. **`backends/nvidia/driver.c`** - Fix 5

### Build System (Required)
3. **`runtime/build.py`** - Fix 4 (Remove -fPIC, add Windows warnings suppression, add Python library paths)

### Optional Fixes
4. **`tools/build_extern.py`** - Fix 6 (only if using build_extern tool)

### AMD Backend (For AMD GPUs)
5. **`backends/amd/driver.py`** - Fix 7 (HIP include directories)
6. **`backends/amd/driver.c`** - Already has Windows compatibility (conditional includes and LoadLibrary)

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
- **Error 1181**: Usually means `nvcuda.lib` not found - fixed by skipping library linking on Windows
- **Error 1104**: Usually means Python library directory not found - fixed by adding Python library directory to linker search path
  - The fix now checks multiple locations including base Python installation (important for pyenv, system Python)
  - For pyenv installations, libraries are typically at: `C:\Users\<user>\.pyenv\pyenv-win\versions\<version>\libs`
  - Verify Python installation has a `libs` directory containing `python<version>.lib`
- **Error 1120**: Usually means unresolved symbols - fixed by explicitly linking against Python library
  - The fix automatically detects and links against `python312.lib`, `python3.lib`, or `python.lib`
  - This ensures Python symbols are resolved during linking
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
- [ ] No deprecated function warnings (strcat)
- [ ] Flash Attention 2.0 loads successfully
- [ ] Model inference works with Flash Attention 2.0

## Status

✅ **All fixes documented** - Use the automated script (`fix_triton_windows.ps1`) to apply all fixes at once.

**Script Coverage:**
- ✅ Fix 1: NVIDIA driver - CUDA library path detection
- ✅ Fix 2: NVIDIA driver - Library name
- ✅ Fix 3: NVIDIA driver - CUDA include directories + Skip library linking on Windows
- ✅ Fix 4: Build system - Remove -fPIC flag, suppress warnings, add Python library paths
- ✅ Fix 5: NVIDIA driver.c - Windows compatibility (macro fix)
- ✅ Fix 6: Build extern tool - Grep command
- ✅ Fix 7: AMD driver - HIP include directories

**Note**: The script now handles both NVIDIA and AMD backend fixes. AMD driver.c already has Windows compatibility, but the script adds HIP include directory detection.

**Additional Fixes Applied:**
- **Skip library linking on Windows**: Prevents linker error 1181 (nvcuda.lib not found)
- **Python library directory**: Prevents linker error 1104 (Python libraries not found)
- **Warning suppression**: Suppresses deprecated `strcat` warnings on Windows

