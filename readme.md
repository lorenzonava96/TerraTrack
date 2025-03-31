<p align="left">
  <img src="figures/logo.png" alt="TerraTrack" width="160"/>
</p>

# 🛰️ TerraTrack

**TerraTrack** is an open-source, cloud-based workflow for detecting and monitoring slow-moving landslides using Sentinel-2 imagery and optical feature tracking (FT).

It is fully reproducible via Google Colab and supports scalable motion analysis using multiple tracking methods, terrain filtering, and time series reconstruction.

---

## 🚀 Features

- Automated Sentinel-2 image acquisition via Earth Engine API
- Multiple feature tracking methods:
  - FFT-based Normalized Cross-Correlation (**NCC**)
  - Phase Cross-Correlation (**PCC**)
  - Dense Optical Flow (Farneback)
- Custom filtering pipeline:
  - Magnitude, angular coherence, PKR/SNR thresholds
  - Slope/aspect-based filtering, DBSCAN clustering
- Time series reconstruction using weighted or midpoint binning
- Export-ready output formats for QGIS and TICOI

---

## 📒 Get Started

### ▶️ Run in Google Colab (Recommended)

Click the badge below to launch:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lorenzonava96/TerraTrack/blob/main/notebooks/TerraTrack_v1.0.ipynb)

---

### 🖥️ Run Locally

1. Clone the repo:
   ```bash
   git clone https://github.com/lorenzonava96/TerraTrack.git
   cd TerraTrack
