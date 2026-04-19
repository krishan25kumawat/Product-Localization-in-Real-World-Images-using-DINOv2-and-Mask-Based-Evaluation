# Product Localization in Real World Images using DINOv2 and Mask Based Evaluation
This project implements a product localization pipeline using DINOv2 embeddings to detect products in real-world images. The model identifies the approximate location of a product by comparing image features with reference images and generates bounding-box-based masks. The predictions are evaluated against ground-truth annotation masks.
#  Product Localization using DINOv2

##  Overview
This project presents a pipeline for **localizing products in real-world review images** using **DINOv2 embeddings**.  

Instead of using traditional object detection models, this approach relies on **feature similarity between reference product images and regions in review images** to identify product locations.

The detected regions are converted into masks and evaluated against **ground-truth annotation masks** using **IoU and mAP metrics**.

---
The dataset consists of multiple product categories, where each product is identified by a unique product ID. For every product, the dataset includes:

• Original Images:
  Clean reference images of the product used to extract visual features.

• Raw Review Images:
  Real-world images uploaded by users (organized by rating folders), where the product appears under varying conditions such as different lighting, angles, and backgrounds.

• Annotated Defective / Non-Defective:
  Ground-truth annotation images where the product region is manually marked (white) and the background is black. These are used for evaluation.

• Generated Outputs:
  For each review image, the pipeline generates:
  - Crop image (localized product region)
  - Overlay image (bounding box visualization)
  - Mask image (binary rectangular mask)

The dataset is hierarchical and not all folders are guaranteed to contain data, so the pipeline dynamically checks for the existence of images and subfolders at every level.
##  Key Features

-  Embedding-based product localization (no training required)
-  GPU-accelerated pipeline (DINOv2)
-  Works in open-world setting (no fixed classes)
-  Product-wise processing for scalability
-  Evaluation using IoU, Precision, Recall, F1-score, and mAP

---

##  Pipeline Workflow

### Step 1: Data Loading
- Load `original_images` (reference product images)
- Load `raw_review_images` (real-world images)

---

### Step 2: Feature Extraction
- Extract embeddings from original images using **DINOv2**
- These embeddings represent the product's visual identity

---

### Step 3: Region Proposal (Sliding Window)
- Each review image is divided into multiple regions
- Regions are generated at multiple scales

---

### Step 4: Similarity Matching
- Each region is encoded using DINOv2
- Compared with original embeddings using cosine similarity
- Best matching region is selected

---

### Step 5: Local Refinement
- Fine-grained search around best region
- Produces tighter bounding box

---

### Step 6: Output Generation
For each review image:
- `crop.jpg` → detected product region  
- `overlay.jpg` → bounding box visualization  
- `mask.png` → binary mask (rectangle)  
- `prediction.json` → bounding box + score  

---

### Step 7: Mask-Based Evaluation
- Predicted mask is compared with annotated masks
- Metrics computed:
  - IoU (Intersection over Union)
  - Precision / Recall / F1-score
  - AP@0.50, AP@0.75
  - mAP@0.50:0.95

---

##  Dataset Structure
data/
├── category/                                                                                                                
│ ├── product_id/                                                                                                            
│ │ ├── original_images/                                                                                                     
│ │ │ ├── 1.jpg                                                                                                              
│ │ │ ├── 2.jpg                                                                                                            
│ │ │                                                                                                                        
│ │ ├── raw_review_images/                                                                                                   
│ │ │ ├── rating_folder/                                                                                                     
│ │ │ │ ├── image.jpg                                                                                                        
│ │ │                                                                                                                        
│ │ ├── annotated_defective/                                                                                                 
│ │ │ ├── rating_folder/                                                                                                     
│ │ │                                                                                                                        
│ │ ├── annotated_non_defective/                                                                                             
│ │ │ ├── rating_folder/                                                                                                     
│ │                                                                                                                          
│ │ ├── comparisons_dinov2_fast/                                                                                             
│ │ ├── image_name/                                                                                                          
│ │ ├── crop.jpg                                                                                                             
│ │ ├── overlay.jpg                                                                                                          
│ │ ├── mask.png                                                                                                             
│ │ ├── prediction.json                                                                                                      



---

##  Evaluation Metrics

- **IoU (Intersection over Union)**  
  Measures overlap between predicted mask and ground truth mask  

- **GT Coverage**  
  Measures how much of the ground truth is captured  

- **Precision / Recall / F1-score**  
  Based on IoU thresholds  

- **mAP (mean Average Precision)**  
  Computed over IoU thresholds from 0.50 to 0.95  

---

##  Tech Stack

- Python
- PyTorch
- DINOv2 (Vision Transformer)
- OpenCV
- NumPy

---

##  Author
Krishan Kumawat
