# Triton Windows Post-Install Fix

## Overview

After building and installing Triton from source on Windows, the Python driver code still contains Linux-specific calls that will fail on Windows. This document describes the required fix to make Triton work on Windows.

## Problem

The `libcuda_dirs()` function` in Triton's NVIDIA driver module attempts to use the Linux-only `/sbin/ldconfig` command to locate CUDA libraries. On Windows, this causes a `FileNotFoundError` when Triton tries to initialize.

**Error:**
```
FileNotFoundError: [WinError 2] The system cannot find the file specified
```

This occurs in `triton\backends\nvidia\driver.py` at line 25 when it tries to execute:
```python
libs = subprocess.check_output(["/sbin/ldconfig", "-p"]).decode(errors="ignore")
```

## Solution

Modify the `libcuda_dirs()` function in the installed Triton package to detect Windows and use Windows-compatible CUDA path detection instead of `ldconfig`.

## Instructions

### Step 1: Locate the Driver File

The file to modify is located at:
```
<venv_path>\Lib\site-packages\triton\backends\nvidia\driver.py
```

For example, if your virtual environment is at `D:\omni\.venv`:
```
D:\omni\.venv\Lib\site-packages\triton\backends\nvidia\driver.py
```

### Step 2: Backup the Original File (Optional but Recommended)

```powershell
Copy-Item "D:\omni\.venv\Lib\site-packages\triton\backends\nvidia\driver.py" "D:\omni\.venv\Lib\site-packages\triton\backends\nvidia\driver.py.backup"
```

### Step 3: Apply the Fix

Open `driver.py` and locate the `libcuda_dirs()` function (around line 20-41).

**Find this code:**
```python
@functools.lru_cache()
def libcuda_dirs():
    if env_libcuda_path := knobs.nvidia.libcuda_path:
        return [env_libcuda_path]

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

**Replace it with:**
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

### Step 4: Verify the Fix

After making the change, verify that:
1. The file saves correctly
2. Python syntax is valid (no indentation errors)
3. The `import platform` statement is present (it should be, but verify it's not causing issues)

### Step 5: Test

Run your Python script that uses Triton/Flash Attention 2. The `FileNotFoundError` should no longer occur.

## What the Fix Does

1. **Detects Windows**: Uses `platform.system() == "Windows"` to detect Windows OS
2. **Searches Windows CUDA Paths**: Looks for CUDA in:
   - `CUDA_PATH` environment variable
   - `CUDA_HOME` environment variable  
   - Default installation path: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA`
   - Versioned subdirectories (e.g., `v12.9`, `v12.8`)
3. **Checks Common Library Directories**: Searches in `bin`, `lib`, `lib64`, and `lib\x64` subdirectories
4. **Preserves Linux Behavior**: The original Linux `ldconfig` code is preserved for non-Windows systems
5. **Returns Paths**: Returns a list of unique CUDA library directories found on Windows

## Important Notes

- **This fix is required** even if you've compiled Triton from source with all Windows compatibility fixes
- The fix only affects the Python driver code, not the compiled C++ components
- **Reapply after reinstalling Triton**: If you reinstall or update Triton, you'll need to apply this fix again
- The fix is **backward compatible** - it preserves the original Linux behavior for non-Windows systems

## Alternative: Automated Script

You can create a PowerShell script to automate this fix:

```powershell
# fix_triton_windows.ps1
$driverFile = "$env:VIRTUAL_ENV\Lib\site-packages\triton\backends\nvidia\driver.py"

if (-not (Test-Path $driverFile)) {
    Write-Host "Error: Triton driver file not found at $driverFile"
    exit 1
}

# Backup
Copy-Item $driverFile "$driverFile.backup"

# Read file
$content = Get-Content $driverFile -Raw

# Check if already fixed
if ($content -match "Windows-compatible CUDA library detection") {
    Write-Host "Fix already applied!"
    exit 0
}

# Apply fix (you would need to implement the replacement logic here)
# This is a simplified example - you'd need to do proper Python code replacement

Write-Host "Fix applied successfully!"
```

## Troubleshooting

### Fix Not Working

1. **Verify the file was modified**: Check that the Windows detection code is present
2. **Check Python cache**: Clear `__pycache__` directories:
   ```powershell
   Remove-Item -Recurse -Force "$env:VIRTUAL_ENV\Lib\site-packages\triton\backends\nvidia\__pycache__"
   ```
3. **Verify CUDA paths**: Ensure CUDA is installed and `CUDA_PATH` is set (if using environment variable)

### Still Getting Errors

- Verify CUDA Toolkit is installed
- Check that `CUDA_PATH` environment variable points to your CUDA installation
- Ensure the CUDA installation path exists and contains `bin`, `lib`, or `lib64` directories

## Summary

**File to modify:** `<venv>\Lib\site-packages\triton\backends\nvidia\driver.py`  
**Function to modify:** `libcuda_dirs()` (around line 20-41)  
**Change:** Add Windows detection and CUDA path search before the `ldconfig` call  
**Result:** Triton will work on Windows without `FileNotFoundError`

