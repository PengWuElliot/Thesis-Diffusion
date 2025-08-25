<div align="center">

<h2><strong>Visual-based anomaly detection empowered by synthetic image generation</strong></h2>  
<strong>Anomaly inpainting for overhead cranes</strong>

</div>

<p align="center">
  <img src="generated.png" width="75%" alt="Teaser image for AnomalyAny">
</p>

---

> **Abstract:**  Detecting anomalies in the manufacturing domain presents significant challenges due to the scarcity of defective samples. Generative models provide a promising solution by synthesizing realistic training data. However, most anomaly generation approaches rely exclusively on standard benchmark datasets and lack validation in real-world industrial settings. Given the wide variability in the type and appearance of anomalies in industrial settings, generated images should support downstream model training and also closely resemble actual defects. In this work, we propose a framework for anomaly generation and detection, using overhead cranes as a case study. We construct a custom dataset of overhead cranes, including both normal and defective images, with various anomaly types such as surface abrasion, oil stains, and rust. Our approach adopts an inpainting-based diffusion model, trained on our overhead crane dataset. This method enables the generation of either small-scale or large-scale anomalies with a few training samples. Experimental results demonstrate that the generated anomalies closely resemble real crane defects and enhance the performance of downstream anomaly detection models. GitHub Repository: https://github.com/PengWuElliot/Thesis-Diffusion.
---


## 💻 Requirements
Python 3.7+
CUDA 11.6+

## 📦 Installation
```bash
mamba env create -f env.yml
```

## 🖼️ Anomaly Inference
### 🔧 1. Train the Anomaly Inpainting Model

- Download the crane dataset from: `https://drive.google.com/drive/folders/1O-52sTPgGEZmAVGDsCh7yaoS6fTpkE3Q?usp=sharing` 
- Update the dataset path in the configuration file 
- Run the training script:

  ```bash
  sbatch run.slurm
  
### 🔧 2. Generate new images
- Download the crane dataset from: `https://drive.google.com/drive/folders/1T0NvriRwDau4DBF_8a9b4AsTXgJvEeCu?usp=sharing` 
- Update the dataset path in the configuration file 
- Run the inference script:

 ```bash
  sbatch inference.slurm
