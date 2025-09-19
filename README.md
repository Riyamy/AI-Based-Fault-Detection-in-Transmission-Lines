# AI-Based Fault Detection in Transmission Lines

This project applies **MATLAB/Simulink** and **Machine Learning** for detecting and classifying faults in transmission lines.  
It combines **signal processing (FFT & Wavelet)** with **ML models** to build a predictive maintenance solution.

---

## Objectives
- Simulate transmission line faults using MATLAB/Simulink  
- Generate a synthetic dataset of different fault types  
- Extract features using FFT and Wavelet transforms  
- Train ML models (Random Forest, SVM, Neural Networks)  
- Build a simple AI pipeline for automated fault detection  

---

## Tools & Technologies
- **MATLAB/Simulink** – fault signal generation  
- **Python (Scikit-learn, Pandas, NumPy)** – ML modeling  
- **FFT, Wavelet** – feature extraction  
- **Matplotlib, Seaborn** – visualization  

---

## How to Run & Results
```bash
1. Generate fault signals in MATLAB and save as .mat files in data/raw/

2. Extract features
python preprocessing/prepare_features.py

3. Train model
python models/train_and_evaluate.py

4. Run inference on new signals
python pipeline/infer.py


✅ Results
 - >99% classification accuracy
 - Early fault detection for predictive maintenance
 - Reliable system for power transmission monitoring
