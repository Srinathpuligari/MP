"""
Dataset Download and Preparation Utilities
"""

import os
import requests
from tqdm import tqdm


DATASETS = {
    "polyu": {
        "name": "PolyU Finger Knuckle Print Database",
        "url": "http://www4.comp.polyu.edu.hk/~biometrics/FKP.htm",
        "requires_application": True,
        "description": "7,920 images from 165 subjects. Requires application approval (1-2 days)."
    },
    "iitd": {
        "name": "IIT Delhi Touchless Palmprint",
        "url": "https://www4.comp.polyu.edu.hk/~csajaykr/IITD/Database_Palm.htm",
        "requires_application": False,
        "description": "2,400 images from 230 subjects. Free download."
    },
    "polyu_3d": {
        "name": "PolyU 3D Palmprint Database",
        "url": "http://www4.comp.polyu.edu.hk/~biometrics/3D_Palmprint.htm",
        "requires_application": True,
        "description": "8,000 3D samples from 400 subjects. Best for 3D training!"
    }
}


def show_datasets():
    """Display available datasets"""
    print("\n" + "="*70)
    print("📊 Available Knuckle/Palmprint Datasets")
    print("="*70)
    
    for key, info in DATASETS.items():
        print(f"\n🔹 {info['name']}")
        print(f"   URL: {info['url']}")
        print(f"   Application Required: {'Yes' if info['requires_application'] else 'No'}")
        print(f"   Description: {info['description']}")
    
    print("\n" + "="*70)
    print("\n📝 How to Get Datasets:")
    print("\n1. PolyU FKP (Recommended):")
    print("   - Visit: http://www4.comp.polyu.edu.hk/~biometrics/FKP.htm")
    print("   - Click 'Request Database'")
    print("   - Fill form (Name, Email, Organization, Purpose)")
    print("   - Wait 1-2 days for approval")
    print("   - Download when approved")
    
    print("\n2. IIT Delhi (Instant - No Application):")
    print("   - Visit: https://www4.comp.polyu.edu.hk/~csajaykr/IITD/Database_Palm.htm")
    print("   - Download directly")
    
    print("\n3. PolyU 3D (Best for 3D DGCNN):")
    print("   - Visit: http://www4.comp.polyu.edu.hk/~biometrics/3D_Palmprint.htm")
    print("   - Apply similar to PolyU FKP")
    
    print("\n" + "="*70)


def download_file(url, destination, filename=None):
    """
    Download file with progress bar
    """
    if filename is None:
        filename = url.split('/')[-1]
    
    filepath = os.path.join(destination, filename)
    
    print(f"Downloading {filename}...")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f, tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"✅ Downloaded: {filepath}")
        return filepath
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None


def create_dataset_structure():
    """Create folder structure for datasets"""
    base_dir = os.path.dirname(__file__)
    
    folders = [
        os.path.join(base_dir, 'raw'),
        os.path.join(base_dir, 'processed'),
        os.path.join(base_dir, 'train'),
        os.path.join(base_dir, 'val'),
        os.path.join(base_dir, 'test')
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"✅ Created: {folder}")
    
    print("\n📁 Dataset folder structure ready!")
    print("\nNext steps:")
    print("1. Download dataset manually and extract to 'raw/' folder")
    print("2. Run: python dataset/prepare_data.py")


if __name__ == "__main__":
    print("="*70)
    print("🔧 Dataset Download Utility")
    print("="*70)
    
    show_datasets()
    
    print("\n" + "="*70)
    print("Setting up dataset folders...")
    print("="*70)
    
    create_dataset_structure()
    
    print("\n" + "="*70)
    print("⚠️  MANUAL DOWNLOAD REQUIRED")
    print("="*70)
    print("\nMost knuckle datasets require manual application.")
    print("Please visit the URLs above to request access.")
    print("\nOnce downloaded:")
    print("  1. Extract files to: testingdgcnn/dataset/raw/")
    print("  2. Run: python dataset/prepare_data.py")
    print("="*70)
