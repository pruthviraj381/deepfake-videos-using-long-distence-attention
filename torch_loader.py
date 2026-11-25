"""
Central PyTorch loader with graceful DLL failure diagnostics for Windows.
Attempts import exactly once per process to prevent PyTorch re-init errors.
"""

import sys

torch = None
nn = None
F = None
Dataset = object
DataLoader = None
TORCH_AVAILABLE = False
TORCH_ERROR = None

try:
    import torch as _torch
    import torch.nn as _nn
    import torch.nn.functional as _F
    from torch.utils.data import Dataset as _Dataset, DataLoader as _DataLoader

    torch = _torch
    nn = _nn
    F = _F
    Dataset = _Dataset
    DataLoader = _DataLoader
    TORCH_AVAILABLE = True
except (OSError, ImportError) as e:
    TORCH_ERROR = str(e)
    # Print clear, user-friendly instructions to terminal on Windows
    print("\n" + "!" * 75)
    print("  [PyTorch DLL Notice on Windows]")
    print(f"  Error details: {e}")
    print("  Cause: PyTorch requires the Microsoft Visual C++ 2015-2022 Redistributable (x64).")
    print("  Solution:")
    print("    Double-click the installer in your Downloads folder:")
    print(r"    C:\Users\RRC\Downloads\vc_redist.x64.exe")
    print("    Click 'Install' or 'Repair' (takes ~5 seconds), then restart app.py.")
    print("!" * 75 + "\n")
