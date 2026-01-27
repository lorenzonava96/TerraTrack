<p align="left">
  <img src="figures/logo.png" alt="TerraTrack" width="160"/>
</p>

🛰️ **TerraTrack** is an open-source, cloud-based workflow for detecting and monitoring slow-moving landslides using Sentinel-2 imagery and optical feature tracking. For further details, please consult the paper "A Workflow to Identify and Monitor Slow-Moving Landslides through Spaceborne Optical Feature Tracking" by [Nava et al (2025)]([https://doi.org/10.5194/gmd-19-167-2026](https://doi.org/10.5194/egusphere-2025-2795)).


---

## 📒 Get Started

### ▶️ Run in Google Colab (Recommended)

Click the badge below to launch:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lorenzonava96/TerraTrack/blob/main/notebooks/TerraTrack_v1.ipynb)

---

### 🖥️ Run Locally

```bash
# Clone the repository
git clone https://github.com/lorenzonava96/TerraTrack.git
cd TerraTrack

# Create a new conda environment
conda create --name terratrack_env python=3.10
conda activate terratrack_env

# Install required dependencies
pip install -r requirements.txt

# Move the notebook to the root directory so it can access the src/ folder
mv notebooks/TerraTrack_v1_local.ipynb

```
Launch the notebook using Jupyter Notebook or another compatible environment

---

## Workflow

<p align="left">
  <img src="figures/Workflow2.png" alt="Workflow" width="2000"/>
</p>

- Automated Sentinel-2 image acquisition via Earth Engine API
- Multiple feature tracking methods:
  - FFT-based Normalized Cross-Correlation (**FFT-NCC**)
  - Phase Cross-Correlation (**PCC**)
  - Median Dense Optical Flow (Farneback)
- Custom filtering pipeline:
  - Magnitude, angular coherence, PKR/SNR thresholds
  - Slope/aspect-based filtering, clustering
- Time series reconstruction using weighted or midpoint binning
- Export-ready, georeferenced median velocity maps and displacement time series, compatible with InSAR Explorer in QGIS.

## Repository Structure
```bash

TerraTrack/
├── notebooks/           # Main notebooks
├── src/                 # Helpers
├── figures/             # Logo, visuals
├── requirements.txt     # Dependencies for Colab
├── LICENSE
└── README.md

```
## License

This project is licensed under the [MIT License](LICENSE).

## Citation

A peer-reviewed paper describing TerraTrack is currently in preparation. Citation details will be provided once available.

## Feedback & Support

Have questions, suggestions, or found a bug? Feel free to [open an issue](https://github.com/lorenzonava96/TerraTrack/issues).


