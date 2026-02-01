"""
Automatic Environment Setup for DGCNN Testing
Installs all required dependencies
"""

import subprocess
import sys
import os

def install_requirements():
    """Install all packages from requirements.txt"""
    print("="*60)
    print("🚀 Setting up DGCNN Testing Environment")
    print("="*60)
    
    req_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    
    if not os.path.exists(req_file):
        print("❌ Error: requirements.txt not found!")
        return False
    
    print("\n📦 Installing PyTorch and dependencies...")
    print("⏱️  This may take 5-10 minutes...\n")
    
    try:
        # Install PyTorch CPU version
        print("Installing PyTorch (CPU version)...")
        subprocess.check_call([
            sys.executable, 
            "-m", 
            "pip", 
            "install", 
            "torch==2.0.1", 
            "torchvision==0.15.2",
            "--index-url",
            "https://download.pytorch.org/whl/cpu"
        ])
        
        # Install other requirements
        print("\nInstalling other dependencies...")
        subprocess.check_call([
            sys.executable, 
            "-m", 
            "pip", 
            "install", 
            "-r", 
            req_file
        ])
        
        print("\n" + "="*60)
        print("✅ Environment setup complete!")
        print("="*60)
        
        # Test imports
        print("\n🧪 Testing imports...")
        test_imports()
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Installation failed: {e}")
        return False


def test_imports():
    """Test if all packages are installed correctly"""
    packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('open3d', 'Open3D'),
        ('sklearn', 'Scikit-learn')
    ]
    
    all_ok = True
    for pkg, name in packages:
        try:
            __import__(pkg)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - Failed to import")
            all_ok = False
    
    if all_ok:
        print("\n🎉 All packages installed successfully!")
        print("\n📖 Next steps:")
        print("   1. Download dataset: python dataset/download_dataset.py")
        print("   2. Or manually get PolyU dataset from:")
        print("      http://www4.comp.polyu.edu.hk/~biometrics/FKP.htm")
    else:
        print("\n⚠️  Some packages failed - try running again")


if __name__ == "__main__":
    success = install_requirements()
    
    if not success:
        sys.exit(1)
