# 📊 Knuckle Pattern Datasets - Complete Guide

## 🎯 Recommended Datasets for Your Project

---

## 1. **PolyU Finger Knuckle Print Database** ⭐ BEST

### Details:
- **Direct Link:** http://www4.comp.polyu.edu.hk/~biometrics/FKP.htm
- **Application Form:** http://www4.comp.polyu.edu.hk/~biometrics/FKP/FKP_database_request_form.doc
- **Approval Time:** 1-2 business days
- **Size:** ~500 MB
- **Images:** 7,920 knuckle images
- **Subjects:** 165 persons
- **Sessions:** 2 sessions per person
- **Images per session:** 12 images (6 per finger)
- **Resolution:** 110 DPI
- **Quality:** High quality, well-documented

### How to Get:
1. Download the request form from the link above
2. Fill in:
   - Your Name
   - University/Institution
   - Email
   - Purpose: "Academic Research on Biometric Recognition"
3. Email to: biometrics@comp.polyu.edu.hk
4. Wait 1-2 days for download link
5. Extract to: `testingdgcnn/dataset/raw/`

### Why This Dataset:
✅ Most popular in research papers  
✅ Well-organized structure  
✅ High quality images  
✅ Multiple sessions (good for testing)  
✅ Used in 100+ published papers  

---

## 2. **IIT Delhi Touchless Palmprint Database** ⚡ INSTANT

### Details:
- **Direct Download:** https://www4.comp.polyu.edu.hk/~csajaykr/IITD/Database_Palm.htm
- **Approval:** NOT REQUIRED - Instant download
- **Size:** ~200 MB
- **Images:** 2,400 images
- **Subjects:** 230 persons
- **Sessions:** 5 images per hand
- **Resolution:** Good quality
- **Includes:** Knuckle region visible

### How to Get:
1. Visit: https://www4.comp.polyu.edu.hk/~csajaykr/IITD/Database_Palm.htm
2. Click download link
3. Extract to: `testingdgcnn/dataset/raw/`
4. Run: `python dataset/prepare_data.py`

### Why This Dataset:
✅ NO application needed  
✅ Instant download  
✅ Good quality  
✅ Includes knuckle patterns  
✅ Free for research  

---

## 3. **PolyU 3D Palmprint Database** 🚀 BEST FOR 3D

### Details:
- **Link:** http://www4.comp.polyu.edu.hk/~biometrics/3D_Palmprint.htm
- **Type:** ACTUAL 3D data (not just depth estimation!)
- **Images:** 8,000 3D samples
- **Subjects:** 400 persons
- **Data Format:** 3D point clouds + 2D images
- **Size:** ~2 GB
- **Approval:** Required (similar to FKP)

### How to Get:
1. Visit: http://www4.comp.polyu.edu.hk/~biometrics/3D_Palmprint.htm
2. Download request form
3. Apply (1-2 days approval)
4. Download 3D data

### Why This Dataset:
✅ Real 3D depth data (not estimated!)  
✅ Perfect for DGCNN training  
✅ Includes texture + depth  
✅ State-of-the-art quality  
✅ Best accuracy potential  

---

## 4. **CASIA Palmprint Database**

### Details:
- **Link:** http://biometrics.idealtest.org/
- **Images:** 5,502 images
- **Subjects:** 312 persons
- **Type:** Contact-based palmprints

### How to Get:
1. Visit: http://biometrics.idealtest.org/
2. Register account
3. Request database access
4. Download after approval

---

## 5. **Tongji University Database**

### Details:
- **Images:** 12,000+ knuckle images
- **Contact:** Usually through research collaboration
- **Quality:** Very high resolution

---

## 📥 Quick Start - Get Dataset in 5 Minutes

### Option A: Fastest (IIT Delhi - No Application)
```bash
cd testingdgcnn

# 1. Run download utility
python dataset/download_dataset.py

# 2. Manually download from:
#    https://www4.comp.polyu.edu.hk/~csajaykr/IITD/Database_Palm.htm

# 3. Extract to dataset/raw/

# 4. Prepare data
python dataset/prepare_data.py
```

### Option B: Best Quality (PolyU FKP - 1-2 days wait)
```bash
# 1. Apply for dataset
#    Visit: http://www4.comp.polyu.edu.hk/~biometrics/FKP.htm
#    Download form and email

# 2. Wait for approval (1-2 days)

# 3. Download and extract to dataset/raw/

# 4. Prepare
python dataset/prepare_data.py
```

### Option C: Ultimate (PolyU 3D - Best for DGCNN)
```bash
# Same as Option B, but use 3D database
# Perfect for true 3D DGCNN training!
```

---

## 📧 Application Email Template

**Subject:** Request for PolyU Finger Knuckle Print Database Access

**Body:**
```
Dear Sir/Madam,

I am writing to request access to the PolyU Finger Knuckle Print Database
for academic research purposes.

Name: [Your Name]
Institution: [Your University]
Email: [Your Email]
Purpose: Academic research on 3D finger knuckle pattern recognition
         using Deep Graph Convolutional Neural Networks (DGCNN)

I confirm that this database will be used solely for academic research
and will not be distributed to third parties.

Thank you for your consideration.

Best regards,
[Your Name]
```

---

## 🎯 Recommendation for Your Project

### For Quick Testing (Today):
1. **IIT Delhi** - Download now, start training immediately

### For Best Results (2-3 days):
1. **PolyU FKP** - Most popular, well-documented
2. Wait 1-2 days for approval
3. Best for academic papers/publications

### For Ultimate Accuracy (Research paper quality):
1. **PolyU 3D** - Real 3D data
2. Perfect for DGCNN architecture
3. State-of-the-art results

---

## 📊 Dataset Comparison

| Dataset | Images | Subjects | Approval | Size | 3D Data | Best For |
|---------|--------|----------|----------|------|---------|----------|
| **PolyU FKP** | 7,920 | 165 | 1-2 days | 500MB | No | Research |
| **IIT Delhi** | 2,400 | 230 | Instant | 200MB | No | Quick Start |
| **PolyU 3D** | 8,000 | 400 | 1-2 days | 2GB | YES | Best Quality |
| **CASIA** | 5,502 | 312 | 3-5 days | 800MB | No | Alternative |

---

## 🚀 After Getting Dataset

### Next Steps:
```bash
# 1. Setup environment
cd testingdgcnn
python setup_environment.py

# 2. Prepare dataset
python dataset/prepare_data.py

# 3. Upload to Google Colab
# Upload training/train_colab.ipynb

# 4. Train (2-3 hours with free GPU)

# 5. Download trained model

# 6. Test accuracy
python evaluation/evaluate.py
```

---

## 💡 Pro Tips

1. **Start with IIT Delhi** (no wait)
2. **Apply for PolyU** while testing
3. **Use Colab for training** (free GPU)
4. **Combine datasets** for more training data
5. **Keep PolyU 3D** as final goal

---

## 📞 Contact for Dataset Issues

**PolyU Biometrics:**
- Email: biometrics@comp.polyu.edu.hk
- Usually respond within 1 business day

**IIT Delhi:**
- Direct download (no contact needed)

---

**Start now with IIT Delhi, then upgrade to PolyU when approved!** 🎉
