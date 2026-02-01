# 🧪 DGCNN Testing Environment

**Isolated testing environment for 3D knuckle pattern recognition using DGCNN**

⚠️ This folder is separate from the main project - safe to experiment!

---

## 📊 Knuckle Pattern Datasets

### **Recommended Datasets:**

#### 1. **PolyU Finger Knuckle Print Database** (Most Popular)
- **Link:** http://www4.comp.polyu.edu.hk/~biometrics/FKP.htm
- **Contact:** Apply via form (usually approved in 1-2 days)
- **Details:**
  - 7,920 images
  - 165 subjects
  - 2 sessions
  - 6 images per finger per session
  - 110 DPI resolution
- **Best for:** Research & Training

#### 2. **IIT Delhi Touchless Palmprint Database**
- **Link:** https://www4.comp.polyu.edu.hk/~csajaykr/IITD/Database_Palm.htm
- **Free download** (no application needed)
- **Details:**
  - Contains knuckle region
  - 2,400 images
  - 230 subjects
  - Contact-free acquisition

#### 3. **CASIA Palmprint Database**
- **Link:** http://biometrics.idealtest.org/
- **Details:**
  - 5,502 images
  - 312 subjects
  - Includes knuckle features

#### 4. **PolyU 3D Palmprint Database**
- **Link:** http://www4.comp.polyu.edu.hk/~biometrics/3D_Palmprint.htm
- **Details:**
  - Actual 3D depth data
  - 8,000 samples
  - 400 subjects
  - Perfect for 3D DGCNN training

---

## 🚀 Quick Start

### Step 1: Install Dependencies
```bash
cd testingdgcnn
python setup_environment.py
```

### Step 2: Download Dataset
**Option A - Manual:**
1. Visit: http://www4.comp.polyu.edu.hk/~biometrics/FKP.htm
2. Fill application form
3. Download when approved (1-2 days)
4. Extract to `testingdgcnn/dataset/raw/`

**Option B - Auto (if you have the dataset):**
```bash
python dataset/prepare_data.py
```

### Step 3: Train on Colab (Recommended - FREE GPU)
1. Upload `training/train_colab.ipynb` to Google Colab
2. Upload `models/` folder to Colab
3. Run all cells (2-3 hours with GPU)
4. Download `dgcnn_trained.pth` from Colab

### Step 4: Evaluate
```bash
python evaluation/evaluate.py --weights weights/dgcnn_trained.pth
```

### Step 5: Integrate to Main Project
```bash
python test_integration.py
```

If accuracy > 90%, copy to main:
```bash
copy weights\dgcnn_trained.pth ..\backend\weights\
```

---

## 📁 Folder Structure

```
testingdgcnn/
├── README.md                      # This file
├── requirements.txt               # Dependencies
├── setup_environment.py          # Auto-installer
├── dataset/
│   ├── download_dataset.py       # Dataset utilities
│   ├── prepare_data.py           # Preprocessing
│   └── raw/                      # Put downloaded datasets here
├── models/
│   ├── dgcnn.py                  # DGCNN architecture
│   ├── depth_estimator.py        # 2D → 3D conversion
│   └── pointcloud_processor.py   # 3D point cloud processing
├── training/
│   ├── train_local.py            # CPU training (slow)
│   └── train_colab.ipynb         # GPU training (fast)
├── evaluation/
│   ├── evaluate.py               # Accuracy metrics
│   └── visualize.py              # Visualization
├── weights/                       # Saved models
│   └── .gitkeep
└── test_integration.py           # Test before main project
```

---

## 🎯 Training Options

### Option 1: Google Colab (Recommended - FREE GPU)
- Upload notebook to colab.research.google.com
- Training time: 2-3 hours
- Accuracy: 95%+
- Cost: FREE

### Option 2: Kaggle Notebooks (FREE GPU)
- Upload to kaggle.com/code
- 30 hours/week free GPU
- Same performance as Colab

### Option 3: Local PC (CPU only - Very Slow)
- Training time: 24-48 hours
- Accuracy: Same
- Cost: FREE (but time-consuming)

---

## 📊 Expected Results

| Metric | Target |
|--------|--------|
| Accuracy | 95%+ |
| FAR (False Accept) | <1% |
| FRR (False Reject) | <5% |
| EER (Equal Error) | <3% |
| Inference Time | <2 sec |

---

## 🔄 Integration Steps

Once trained model achieves good accuracy:

1. **Backup current system:**
   ```bash
   cd ..\backend
   copy ai_engine.py ai_engine_resnet_backup.py
   ```

2. **Copy trained weights:**
   ```bash
   copy ..\testingdgcnn\weights\dgcnn_trained.pth weights\
   ```

3. **Update ai_engine.py** to use DGCNN instead of ResNet50

4. **Test** with frontend

5. **Commit** when working

---

## ⚠️ Important Notes

- ✅ This folder is SEPARATE from main project
- ✅ Experiment freely - won't break main system
- ✅ Only integrate when accuracy is good
- ✅ Always backup before integrating

---

## 📧 Dataset Access

**PolyU Database Application:**
- Visit: http://www4.comp.polyu.edu.hk/~biometrics/FKP.htm
- Click "Request Database"
- Fill form with:
  - Your name
  - University/Organization
  - Email
  - Purpose: Academic Research
- Usually approved in 1-2 days
- Check email for download link

---

## 🆘 Troubleshooting

**Issue: Can't download dataset**
- Apply for PolyU (takes 1-2 days)
- OR use IIT Delhi (instant download)
- OR collect your own data

**Issue: Training too slow**
- Use Google Colab (FREE GPU)
- Don't train on laptop CPU

**Issue: Out of memory**
- Reduce batch size in training config
- Use Colab with High RAM

---

**Ready to start!** 🚀
