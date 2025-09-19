# ⚡ AI-Based Fault Detection in Transmission Lines

This project integrates **MATLAB/Simulink simulations**, **signal processing (FFT & Wavelet)**, and **AI models** for automated fault detection in transmission lines.

## 🚀 Features
- Synthetic dataset generation using MATLAB/Simulink
- FFT & Wavelet transforms for feature extraction
- ML models: Random Forest, SVM, Neural Network
- End-to-end automated fault detection pipeline
- Achieved **>99% classification accuracy**

## 🛠 Tech Stack
- MATLAB/Simulink
- Python (NumPy, Pandas, Scikit-learn, TensorFlow/Keras)
- Signal Processing: FFT, PyWavelets
- Visualization: Matplotlib, Seaborn

## 📂 How to Run
1. Run MATLAB script to simulate and export fault data:
   ```bash
   matlab -batch "run('matlab_simulation/fault_simulation.m')"
   ```
2. Preprocess & extract features:
   ```bash
   python preprocessing/fft_wavelet_features.py
   ```
3. Train ML models:
   ```bash
   python models/train_models.py
   ```
4. Run AI pipeline:
   ```bash
   python pipeline/fault_detection_pipeline.py
   ```

---
