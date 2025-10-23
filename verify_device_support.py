#!/usr/bin/env python3
"""
Verification script to check device support (CUDA/MPS/CPU)
"""
import torch
import sys

def verify_device_support():
    """Check and report available devices"""
    print("=" * 60)
    print("Device Support Verification")
    print("=" * 60)
    
    # Check PyTorch version
    print(f"\nPyTorch Version: {torch.__version__}")
    
    # Check CUDA
    print(f"\nCUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  CUDA Device Count: {torch.cuda.device_count()}")
        print(f"  CUDA Device Name: {torch.cuda.get_device_name(0)}")
    
    # Check MPS
    print(f"\nMPS Available: {torch.backends.mps.is_available()}")
    if torch.backends.mps.is_available():
        print(f"  MPS Built: {torch.backends.mps.is_built()}")
    
    # Determine selected device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\n✓ Selected Device: CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"\n✓ Selected Device: MPS")
    else:
        device = torch.device("cpu")
        print(f"\n✓ Selected Device: CPU")
    
    # Test basic operations
    print(f"\n" + "=" * 60)
    print("Testing Basic Operations")
    print("=" * 60)
    
    try:
        # Create a simple tensor
        x = torch.randn(10, 10).to(device)
        print(f"✓ Tensor creation on {device}: Success")
        
        # Test multiplication
        y = torch.randn(10, 10).to(device)
        z = torch.matmul(x, y)
        print(f"✓ Matrix multiplication on {device}: Success")
        
        # Test synchronization
        if device.type == 'cuda':
            torch.cuda.synchronize()
            print(f"✓ CUDA synchronization: Success")
        elif device.type == 'mps':
            torch.mps.synchronize()
            print(f"✓ MPS synchronization: Success")
        
        # Test memory tracking
        if device.type == 'cuda':
            mem = torch.cuda.max_memory_allocated() / 1024 ** 2
            print(f"✓ CUDA memory tracking: {mem:.2f} MB")
        elif device.type == 'mps':
            mem = torch.mps.current_allocated_memory() / 1024 ** 2
            print(f"✓ MPS memory tracking: {mem:.2f} MB")
        
        print(f"\n✓ All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        return False

if __name__ == "__main__":
    success = verify_device_support()
    sys.exit(0 if success else 1)
