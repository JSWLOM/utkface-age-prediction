# UTKFace Age Prediction from RGB Images

## Overview
End-to-end machine learning project that predicts a person's age 
from RGB face images using the UTKFace dataset. The project covers 
image feature extraction, training multiple regression models, 
and comparing their performance.

## Dataset
- **Source:** UTKFace Dataset (Kaggle)
- **Images Used:** 8,000 (from 23,705 total)
- **Image Type:** RGB (3 channels)
- **Age Range:** 1 – 116 years
- **Features per Image:** 1,664
  - 96 Color Histogram features (R, G, B channels)
  - 1,568 HOG (Histogram of Oriented Gradients) features

## Models Trained
| Model | MAE | RMSE | R² Score | Accuracy |
|-------|-----|------|----------|----------|
| Random Forest Regressor | 9.94 yrs | 13.18 yrs | 0.5611 | 56.11% |
| SVR (RBF Kernel) | 8.08 yrs | 10.61 yrs | 0.7154 | 71.54% |
| Neural Network (50 epochs) | 7.46 yrs | 10.52 yrs | 0.7202 | 72.02% |

**Best Model: Neural Network — MAE of 7.46 years, R² of 0.72**

## Project Pipeline
```
RGB Images (UTKFace)
      ↓
Feature Extraction
  ├── Color Histogram (R, G, B channels)
  └── HOG Features (shape & texture)
      ↓
Feature Scaling (StandardScaler)
      ↓
Train/Test Split (80/20)
      ↓
Model Training
  ├── Random Forest Regressor
  ├── SVR (RBF Kernel)
  └── Neural Network (4 layers, 50 epochs)
      ↓
Evaluation & Comparison
```

## Key Results
- Neural Network outperformed traditional ML models on image data
- Average age prediction error of only **7.46 years**
- Model performs best on ages 20–50 (highest training data density)
- RGB color + HOG shape features proven effective for age estimation
  without requiring a CNN architecture

## Tech Stack
- **Python** — core language
- **OpenCV** — image loading and processing
- **Scikit-image** — HOG feature extraction
- **Scikit-learn** — Random Forest, SVR, scaling, evaluation
- **TensorFlow / Keras** — Neural Network training
- **Matplotlib / Seaborn** — visualizations
- **Pandas / NumPy** — data handling

## Project Structure
```
utkface-age-prediction/
│
├── UTKface_prediction.ipynb   # Main notebook
└── README.md                  # Project documentation
```

## Visualizations Included
- Sample face images with age and gender labels
- HOG feature visualization per image
- RGB channel decomposition
- Age distribution across dataset
- Predicted vs Actual age scatter plots (all 3 models)
- Error distribution histograms
- MAE comparison by age group
- Final model comparison bar chart

## Lab Requirements Met
- RGB image dataset (3 channels) ✅
- Dataset size: 8,000 images (5,000–10,000 range) ✅
- Regression task (continuous age prediction) ✅
- Minimum 50 epochs training ✅
- Dataset visualization ✅
- Multiple models trained and compared ✅

## Author
**Om Jaiswal**  
B.Tech Computer Science (Data Science) — Manipal University Jaipur  
GitHub: [@JSWLOM](https://github.com/JSWLOM)  
LinkedIn: [linkedin.com/in/om-jaiswal-1b315126b](https://linkedin.com/in/om-jaiswal-1b315126b)
