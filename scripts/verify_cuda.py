import sys

def verify_environment():
    print("=== Environment Verification ===")
    
    # Python Version
    print(f"Python Version: {sys.version}")
    
    try:
        import torch
        print(f"\nPyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA Device Count: {torch.cuda.device_count()}")
            print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version (PyTorch built with): {torch.version.cuda}")
        else:
            print("⚠️ CUDA is NOT available. PyTorch will run on CPU.")
    except ImportError:
        print("\n⚠️ PyTorch is not installed. Please install it with CUDA support.")

if __name__ == "__main__":
    verify_environment()
