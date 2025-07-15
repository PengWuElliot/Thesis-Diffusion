<div align="center">

<h2><strong>Visual-based anomaly detection empowered by synthetic image generation</strong></h2>  
<strong>Anomaly inpainting for overhead cranes</strong>

</div>

<p align="center">
  <img src="generated.png" width="75%" alt="Teaser image for AnomalyAny">
</p>

---

> **Abstract:**  Detecting anomalies in the manufacturing domain presents significant challenges due to the scarcity of defective samples. Generative models provide a promising solution by synthesizing realistic training data. However, most existing approaches rely exclusively on standard benchmark datasets, lacking validation in real-world industrial settings. Given the wide variability in the type and appearance of anomalies in industrial settings, generated images must not only support model training but also closely resemble actual defects. In this work, we propose a novel framework for fine-grained anomaly generation, using overhead cranes as a case study. We construct a custom dataset of overhead cranes, including both normal and defective images, with various anomaly types such as rust, cracks, foreign objects, and oil stains. Our approach adopts an inpainting-based diffusion model that focuses on generating anomalies exclusively within predefined masked regions. This method enables the generation of realistic small-scale anomalies with minimal training cost. The proposed diffusion models are trained and evaluated on our crane dataset. Experimental results demonstrate that the generated anomalies closely mimic real-world industrial defects and significantly enhance the performance of downstream anomaly detection models.
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

- Download the crane dataset from: `https://drive.google.com/drive/folders/1zdc6s5y1hFtb-2saGTQcxftciL39KDGs?usp=sharing` 
- Update the dataset path in the configuration file 
- Run the training script:

  ```bash
  sbatch run.slurm
  
### 🔧 2. Train the Anomaly Inpainting Model
- Prepare the inference dataset, refer to the format of the dataset "inference", including: \
Original normal images\
Corresponding masks indicating anomaly regions

 ```bash
  sbatch inference.slurm
