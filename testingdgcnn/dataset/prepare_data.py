"""
Prepare Dataset for Training
Organizes raw dataset into train/val/test splits
"""

import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm


def prepare_polyu_dataset(raw_dir, output_dir, train_split=0.7, val_split=0.15):
    """
    Prepare PolyU Finger Knuckle Print dataset
    
    Expected structure of raw_dir:
        raw/
            Person_001/
                session_1/
                    image_01.jpg
                    image_02.jpg
                    ...
                session_2/
                    ...
            Person_002/
            ...
    """
    print("="*60)
    print("📂 Preparing PolyU Dataset")
    print("="*60)
    
    raw_path = Path(raw_dir)
    
    if not raw_path.exists():
        print(f"❌ Raw directory not found: {raw_dir}")
        print("\nPlease:")
        print("  1. Download PolyU dataset")
        print("  2. Extract to: testingdgcnn/dataset/raw/")
        return False
    
    # Find all persons
    persons = sorted([d for d in raw_path.iterdir() if d.is_dir()])
    
    if len(persons) == 0:
        print("❌ No person folders found in raw directory")
        return False
    
    print(f"\n✅ Found {len(persons)} persons")
    
    # Shuffle and split
    random.shuffle(persons)
    
    n_train = int(len(persons) * train_split)
    n_val = int(len(persons) * val_split)
    
    train_persons = persons[:n_train]
    val_persons = persons[n_train:n_train+n_val]
    test_persons = persons[n_train+n_val:]
    
    print(f"   Train: {len(train_persons)} persons")
    print(f"   Val:   {len(val_persons)} persons")
    print(f"   Test:  {len(test_persons)} persons")
    
    # Copy files
    splits = {
        'train': train_persons,
        'val': val_persons,
        'test': test_persons
    }
    
    for split_name, split_persons in splits.items():
        split_dir = Path(output_dir) / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📋 Processing {split_name} set...")
        
        for person_dir in tqdm(split_persons):
            person_name = person_dir.name
            dest_person_dir = split_dir / person_name
            dest_person_dir.mkdir(exist_ok=True)
            
            # Copy all images
            for img_file in person_dir.rglob('*.jpg'):
                dest_file = dest_person_dir / img_file.name
                shutil.copy2(img_file, dest_file)
    
    print("\n✅ Dataset preparation complete!")
    print(f"\nDataset saved to: {output_dir}")
    return True


def prepare_custom_dataset(raw_dir, output_dir):
    """
    Prepare your own collected dataset
    
    Expected structure:
        raw/
            person1/
                img1.jpg
                img2.jpg
            person2/
            ...
    """
    print("="*60)
    print("📂 Preparing Custom Dataset")
    print("="*60)
    
    raw_path = Path(raw_dir)
    
    if not raw_path.exists():
        print(f"❌ Directory not found: {raw_dir}")
        return False
    
    persons = sorted([d for d in raw_path.iterdir() if d.is_dir()])
    
    if len(persons) == 0:
        print("❌ No person folders found")
        print("\nExpected structure:")
        print("  raw/")
        print("    person_001/")
        print("      img1.jpg")
        print("      img2.jpg")
        return False
    
    print(f"\n✅ Found {len(persons)} persons")
    
    # 70-15-15 split
    random.shuffle(persons)
    n_train = int(len(persons) * 0.7)
    n_val = int(len(persons) * 0.15)
    
    splits = {
        'train': persons[:n_train],
        'val': persons[n_train:n_train+n_val],
        'test': persons[n_train+n_val:]
    }
    
    for split_name, split_persons in splits.items():
        split_dir = Path(output_dir) / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📋 {split_name}: {len(split_persons)} persons")
        
        for person_dir in split_persons:
            dest_dir = split_dir / person_dir.name
            shutil.copytree(person_dir, dest_dir, dirs_exist_ok=True)
    
    print("\n✅ Custom dataset prepared!")
    return True


if __name__ == "__main__":
    base_dir = Path(__file__).parent
    raw_dir = base_dir / "raw"
    output_dir = base_dir.parent / "dataset"
    
    print("="*60)
    print("🔧 Dataset Preparation Tool")
    print("="*60)
    
    print("\nChoose dataset type:")
    print("  1. PolyU Finger Knuckle Print")
    print("  2. Custom/Own collected data")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        success = prepare_polyu_dataset(raw_dir, output_dir)
    elif choice == "2":
        success = prepare_custom_dataset(raw_dir, output_dir)
    else:
        print("❌ Invalid choice")
        success = False
    
    if success:
        print("\n" + "="*60)
        print("🎉 Dataset Ready for Training!")
        print("="*60)
        print("\nNext step:")
        print("  Upload training/train_colab.ipynb to Google Colab")
        print("  and start training!")
    else:
        print("\n❌ Dataset preparation failed")
