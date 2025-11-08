# Triton Windows Compatibility Fix Script
# This script automatically applies all Windows compatibility fixes to Triton

param(
    [string]$VenvPath = $env:VIRTUAL_ENV
)

# Colors for output
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

Write-ColorOutput Green "=========================================="
Write-ColorOutput Green "Triton Windows Compatibility Fix Script"
Write-ColorOutput Green "=========================================="
Write-Output ""

# Detect virtual environment
if (-not $VenvPath) {
    $VenvPath = $PSScriptRoot
    if (Test-Path "$PSScriptRoot\.venv") {
        $VenvPath = "$PSScriptRoot\.venv"
    } elseif (Test-Path "$PSScriptRoot\venv") {
        $VenvPath = "$PSScriptRoot\venv"
    }
    
    # Try to find from Python
    try {
        $pythonPath = python -c "import sys; print(sys.prefix)" 2>$null
        if ($pythonPath -and (Test-Path "$pythonPath\Lib\site-packages\triton")) {
            $VenvPath = $pythonPath
        }
    } catch {
        # Ignore
    }
}

$tritonBase = "$VenvPath\Lib\site-packages\triton"
if (-not (Test-Path $tritonBase)) {
    Write-ColorOutput Red "Error: Triton not found at $tritonBase"
    Write-Output "Please specify the virtual environment path with -VenvPath parameter"
    Write-Output "Or activate your virtual environment and set VIRTUAL_ENV"
    exit 1
}

Write-ColorOutput Cyan "Triton path: $tritonBase"
Write-Output ""

# Function to backup file
function Backup-File($FilePath) {
    if (Test-Path $FilePath) {
        $backupPath = "$FilePath.backup"
        if (-not (Test-Path $backupPath)) {
            Copy-Item $FilePath $backupPath
            Write-ColorOutput Yellow "  Backed up: $FilePath"
        }
    }
}

# Function to apply fix with pattern replacement
function Apply-Fix($FilePath, $Description, $FindPattern, $ReplacePattern) {
    if (-not (Test-Path $FilePath)) {
        Write-ColorOutput Red "  Warning: File not found: $FilePath"
        return $false
    }

    Backup-File $FilePath

    $content = Get-Content $FilePath -Raw
    if ($content -match $FindPattern) {
        $content = $content -replace $FindPattern, $ReplacePattern
        Set-Content -Path $FilePath -Value $content -NoNewline
        Write-ColorOutput Green "  ✓ Fixed: $Description"
        return $true
    } else {
        Write-ColorOutput Yellow "  ⊘ Already fixed or pattern not found: $Description"
        return $false
    }
}

# Function to check if fix is already applied
function Test-Fix($FilePath, $TestPattern) {
    if (-not (Test-Path $FilePath)) {
        return $false
    }
    $content = Get-Content $FilePath -Raw
    return $content -match $TestPattern
}

Write-ColorOutput Cyan "Applying fixes..."
Write-Output ""

# Fix 1: NVIDIA driver.py - libcuda_dirs() Windows compatibility
Write-ColorOutput Cyan "Fix 1: NVIDIA driver - CUDA library path detection"
$driverPy = "$tritonBase\backends\nvidia\driver.py"
if (Test-Path $driverPy) {
    Backup-File $driverPy
    $content = Get-Content $driverPy -Raw
    
    # Check if already fixed
    if ($content -match "Windows-compatible CUDA library detection") {
        Write-ColorOutput Yellow "  ⊘ Already fixed: libcuda_dirs()"
    } else {
        # Apply fix
        $oldPattern = '(?s)(@functools\.lru_cache\(\)\s+def libcuda_dirs\(\):.*?libs = subprocess\.check_output\(\["/sbin/ldconfig", "-p"\]\))'
        $newCode = '@functools.lru_cache()
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
        
        # On Windows, we don''t check for libcuda.so.1 (that''s a Linux library name)
        # Instead, we return the CUDA paths where nvcuda.dll would be found
        return unique_paths
    
    # Linux/Unix path: use ldconfig
    libs = subprocess.check_output(["/sbin/ldconfig", "-p"]).decode(errors="ignore")'
        
        if ($content -match $oldPattern) {
            $content = $content -replace [regex]::Escape($matches[0]), $newCode
            Set-Content -Path $driverPy -Value $content -NoNewline
            Write-ColorOutput Green "  ✓ Fixed: libcuda_dirs()"
        } else {
            Write-ColorOutput Yellow "  ⊘ Pattern not found, may already be fixed"
        }
    }
} else {
    Write-ColorOutput Red "  ✗ File not found: $driverPy"
}

# Fix 2: NVIDIA driver.py - Library name
Write-ColorOutput Cyan "Fix 2: NVIDIA driver - Library name (nvcuda vs libcuda.so.1)"
if (Test-Path $driverPy) {
    $content = Get-Content $driverPy -Raw
    if ($content -match "Windows uses nvcuda\.dll") {
        Write-ColorOutput Yellow "  ⊘ Already fixed: Library name"
    } else {
        $oldPattern = "libraries = \['libcuda\.so\.1'\]"
        $newCode = "# Windows uses nvcuda.dll, Linux uses libcuda.so.1`nimport platform`nif platform.system() == `"Windows`":`n    libraries = ['nvcuda']`nelse:`n    libraries = ['libcuda.so.1']"
        if ($content -match $oldPattern) {
            $content = $content -replace $oldPattern, $newCode
            Set-Content -Path $driverPy -Value $content -NoNewline
            Write-ColorOutput Green "  ✓ Fixed: Library name"
        }
    }
}

# Fix 3: NVIDIA driver.py - CUDA include directories
Write-ColorOutput Cyan "Fix 3: NVIDIA driver - CUDA include directories"
if (Test-Path $driverPy) {
    $content = Get-Content $driverPy -Raw
    if ($content -match "def cuda_include_dirs\(\):") {
        Write-ColorOutput Yellow "  ⊘ Already fixed: cuda_include_dirs()"
    } else {
        # Add cuda_include_dirs function after libcuda_dirs
        $insertAfter = '@functools\.lru_cache\(\)\s+def libcuda_dirs\(\):'
        $newFunction = @'

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


'@
        # Find insertion point
        if ($content -match "(@functools\.lru_cache\(\)\s+def library_dirs\(\):)") {
            $content = $content -replace $matches[0], "$newFunction`n`n$($matches[0])"
            Set-Content -Path $driverPy -Value $content -NoNewline
            Write-ColorOutput Green "  ✓ Added: cuda_include_dirs() function"
        }
        
        # Update compile_module_from_src calls
        $content = Get-Content $driverPy -Raw
        if ($content -match "include_dirs=include_dirs,") {
            $content = $content -replace "include_dirs=include_dirs,", "include_dirs=include_dirs + cuda_include_dirs(),"
            Set-Content -Path $driverPy -Value $content -NoNewline
            Write-ColorOutput Green "  ✓ Updated: compile_module_from_src calls"
        } else {
            Write-ColorOutput Yellow "  ⊘ compile_module_from_src calls may already be updated"
        }
    }
}

# Fix 4: build.py - Remove -fPIC flag
Write-ColorOutput Cyan "Fix 4: Build system - Remove -fPIC flag"
$buildPy = "$tritonBase\runtime\build.py"
if (Test-Path $buildPy) {
    Backup-File $buildPy
    $content = Get-Content $buildPy -Raw
    if ($content -match "-fPIC is not supported on Windows MSVC") {
        Write-ColorOutput Yellow "  ⊘ Already fixed: -fPIC flag"
    } else {
        $oldPattern = 'cc_cmd = \[cc, src, "-O3", "-shared", "-fPIC", "-Wno-psabi", "-o", so\]'
        $newCode = @'
    import platform
    cc_cmd = [cc, src, "-O3", "-shared"]
    # -fPIC is not supported on Windows MSVC
    if platform.system() != "Windows":
        cc_cmd.append("-fPIC")
    cc_cmd.append("-Wno-psabi")
    cc_cmd.append("-o")
    cc_cmd.append(so)
'@
        if ($content -match $oldPattern) {
            $content = $content -replace [regex]::Escape($oldPattern), $newCode
            Set-Content -Path $buildPy -Value $content -NoNewline
            Write-ColorOutput Green "  ✓ Fixed: -fPIC flag removal"
        } else {
            Write-ColorOutput Yellow "  ⊘ Pattern not found, may already be fixed"
        }
    }
} else {
    Write-ColorOutput Red "  ✗ File not found: $buildPy"
}

# Fix 5: driver.c - Windows compatibility
Write-ColorOutput Cyan "Fix 5: NVIDIA driver.c - Windows compatibility"
$driverC = "$tritonBase\backends\nvidia\driver.c"
if (Test-Path $driverC) {
    Backup-File $driverC
    $content = Get-Content $driverC -Raw
    
    # Check if already fixed
    if ($content -match "#ifdef _WIN32\s+#define defineGetFunctionHandle") {
        Write-ColorOutput Yellow "  ⊘ Already fixed: defineGetFunctionHandle macro"
    } else {
        # Fix the macro - replace single macro with conditional compilation
        $oldPattern = '(?s)(#define defineGetFunctionHandle\(name, symbolName\).*?#ifdef _WIN32.*?#else.*?#endif.*?return funcHandle;.*?\s+})'
        $newCode = @'
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
'@
        
        # Try to find and replace the problematic macro
        if ($content -match '(?s)(#define defineGetFunctionHandle\(name, symbolName\).*?/\* Open the shared library \*/\s+#ifdef _WIN32)') {
            # This is the broken version - replace it
            $brokenMacro = $matches[0]
            # Find the end of the macro
            if ($content -match "(?s)$([regex]::Escape($brokenMacro)).*?(return funcHandle;\s+})") {
                $fullMacro = $matches[0]
                $content = $content -replace [regex]::Escape($fullMacro), $newCode
                Set-Content -Path $driverC -Value $content -NoNewline
                Write-ColorOutput Green "  ✓ Fixed: defineGetFunctionHandle macro"
            } else {
                Write-ColorOutput Yellow "  ⚠ Could not auto-fix macro - manual fix required"
                Write-Output "    See TRITON_FLASH_ATTEN_FIX.md Fix 5 for instructions"
            }
        } else {
            Write-ColorOutput Yellow "  ⊘ Pattern not found - may already be fixed or structure differs"
        }
    }
    
    # Check conditional includes
    if ($content -match '#include "cuda\.h"\s+#ifdef _WIN32\s+#include <windows\.h>') {
        Write-ColorOutput Yellow "  ⊘ Already fixed: Conditional includes"
    } else {
        if ($content -match '#include "cuda\.h"\s+#include <dlfcn\.h>') {
            $content = $content -replace '#include "cuda\.h"\s+#include <dlfcn\.h>', '#include "cuda.h"`n#ifdef _WIN32`n#include <windows.h>`n#else`n#include <dlfcn.h>`n#endif'
            Set-Content -Path $driverC -Value $content -NoNewline
            Write-ColorOutput Green "  ✓ Fixed: Conditional includes"
        }
    }
} else {
    Write-ColorOutput Red "  ✗ File not found: $driverC"
}

# Fix 6: build_extern.py - Grep command (optional)
Write-ColorOutput Cyan "Fix 6: Build extern tool - Grep command (optional)"
$buildExternPy = "$tritonBase\tools\build_extern.py"
if (Test-Path $buildExternPy) {
    Backup-File $buildExternPy
    $content = Get-Content $buildExternPy -Raw
    
    # Add platform import if not present
    if (-not ($content -match "import platform")) {
        if ($content -match "(import argparse\s+import subprocess)") {
            $content = $content -replace $matches[0], "$($matches[0])`nimport platform"
            Set-Content -Path $buildExternPy -Value $content -NoNewline
            Write-ColorOutput Green "  ✓ Added: platform import"
        }
    }
    
    # Fix parse_symbols method
    if ($content -match "Windows-compatible: use Python file reading") {
        Write-ColorOutput Yellow "  ⊘ Already fixed: parse_symbols()"
    } else {
        $oldPattern = '(?s)(def parse_symbols\(self, input_file\) -> None:.*?output = subprocess\.check_output\(\["grep", "define", input_file\]\)\.decode\(\)\.splitlines\(\))'
        $newCode = @'
    def parse_symbols(self, input_file) -> None:
        if len(self.symbols) > 0:
            return
        # Windows-compatible: use Python file reading instead of grep
        if platform.system() == "Windows":
            with open(input_file, 'r', encoding='utf-8') as f:
                output = [line for line in f if "define" in line]
        else:
            output = subprocess.check_output(["grep", "define", input_file]).decode().splitlines()
'@
        if ($content -match $oldPattern) {
            $content = $content -replace $matches[0], $newCode
            Set-Content -Path $buildExternPy -Value $content -NoNewline
            Write-ColorOutput Green "  ✓ Fixed: parse_symbols()"
        } else {
            Write-ColorOutput Yellow "  ⊘ Pattern not found, may already be fixed"
        }
    }
} else {
    Write-ColorOutput Yellow "  ⊘ File not found (optional fix): $buildExternPy"
}

# Fix 7: AMD driver.py - HIP include directories
Write-ColorOutput Cyan "Fix 7: AMD driver - HIP include directories"
$amdDriverPy = "$tritonBase\backends\amd\driver.py"
if (Test-Path $amdDriverPy) {
    Backup-File $amdDriverPy
    $content = Get-Content $amdDriverPy -Raw
    
    # Check if already fixed
    if ($content -match "def hip_include_dirs\(\):") {
        Write-ColorOutput Yellow "  ⊘ Already fixed: hip_include_dirs()"
    } else {
        # Add hip_include_dirs function after include_dirs
        $newFunction = @'

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


'@
        # Find insertion point after include_dirs
        if ($content -match "(include_dirs = \[os\.path\.join\(dirname, `"include`"\)\])") {
            $content = $content -replace $matches[0], "$($matches[0])`n`n$newFunction"
            Set-Content -Path $amdDriverPy -Value $content -NoNewline
            Write-ColorOutput Green "  ✓ Added: hip_include_dirs() function"
        }
        
        # Update compile_module_from_src calls
        $content = Get-Content $amdDriverPy -Raw
        if ($content -match "include_dirs=include_dirs,") {
            $content = $content -replace "include_dirs=include_dirs,", "include_dirs=include_dirs + hip_include_dirs(),"
            Set-Content -Path $amdDriverPy -Value $content -NoNewline
            Write-ColorOutput Green "  ✓ Updated: compile_module_from_src calls for AMD"
        } else {
            Write-ColorOutput Yellow "  ⊘ compile_module_from_src calls may already be updated"
        }
    }
} else {
    Write-ColorOutput Yellow "  ⊘ File not found (AMD backend not installed): $amdDriverPy"
}

Write-Output ""
Write-ColorOutput Cyan "Clearing Python cache..."
$cacheDirs = Get-ChildItem -Path $tritonBase -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue
foreach ($dir in $cacheDirs) {
    Remove-Item -Path $dir.FullName -Recurse -Force -ErrorAction SilentlyContinue
}
Write-ColorOutput Green "  ✓ Cleared Python cache"

Write-Output ""
Write-ColorOutput Green "=========================================="
Write-ColorOutput Green "Fix script completed!"
Write-ColorOutput Green "=========================================="
Write-Output ""
Write-Output "Next steps:"
Write-Output "1. Test your script with Flash Attention 2.0"
Write-Output "2. If you encounter issues with driver.c, see TRITON_FLASH_ATTEN_FIX.md"
Write-Output "3. All original files have been backed up with .backup extension"
Write-Output "4. Both NVIDIA and AMD backend fixes have been applied"
Write-Output ""

