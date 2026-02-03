# DiTing2.0-seismic-classification
This repository contains the self-developed source code and benchmarking suite for the research project focused on seismic event classification using the DiTing 2.0 dataset. It provides a complete pipeline from raw waveform processing to multi-class classification and automated system prototyping.

## 🌟 Key Features
* DiTing 2.0 Benchmarking: A dedicated DiTing2.0_benchmark_runner for high-performance 3-class (Earthquake, Explosion, Collapse) and 4-class (including Noise) classification.
* Automated Prototype System: A functional Seismic Event Classification System validated on the MSH dataset, designed for real-world application testing.
* Specialized Data Pipelines: Tailored pre-processing scripts (.ipynb) for processing natural earthquakes, collapses, explosions, and noise events specifically for the DiTing 2.0 data format.
* Extensive Model Supplement: An Additional_code directory containing alternative model architectures and experimental frameworks explored throughout the research.

## 📂 Project Structure

```text
DiTing2.0-seismic-classification/
├── DiTing2.0_benchmark_runner/      # Core Benchmarking Suite for DiTing 2.0
│   ├── models/                      # Model architectures and frameworks
│   ├── experiments/                 # Experimental results and outputs
│   ├── *.py / *.sh                  # Training scripts and terminal execution files
│   └── *.txt                        # Log files and configuration details
├── Seismic Event Classification System/ # Automated Prototype System
│   ├── GUI/                         # Graphical User Interface source files
│   ├── models/                      # Pre-trained model weights storage
│   ├── Interface_Image_Example/     # System demonstration and UI screenshots
│   ├── main.py                      # Main entry point for the prototype
│   ├── utils.py / GUI_ui.py         # Utility functions and UI logic
│   └── events.pt / GUI.ui           # Model checkpoints and UI design files
├── Additional_code/                 # Technical supplement: alternative models and frameworks
├── EQ_EP_SS_data_set_processing.ipynb # Pre-processing for DiTing 2.0 natural events (EQ, EP, SS)
├── noise_dataset_process.ipynb      # Pre-processing for DiTing 2.0 noise samples
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
└── LICENSE                          # MIT License
```
## 📊 Data Availability
To ensure research reproducibility while respecting data provider policies, the datasets used in this project are not hosted in this repository. Please obtain them from the following official sources:
1. DiTing 2.0 Dataset
Used for the core benchmarking experiments (3-class and 4-class classification).
Source: National Earthquake Data Center (CENC)
Access: Requires official registration and application.
Link: https://data.earthquake.cn
2. MSH Dataset
Used for the functional validation of the Seismic Event Classification System prototype.
Source: Zenodo Open Repository
Access: Publicly available for download.
Link: https://zenodo.org/records/5115136

## 🚀 Usage Guide

### Step 1: Dataset Acquisition
The very first step is to obtain the required datasets from their respective official sources:

* **DiTing 2.0**: Submit an official application at [National Earthquake Data Center](https://data.earthquake.cn). This dataset is mandatory for running the benchmark experiments.  
* **MSH Dataset**: Download the dataset from [Zenodo](https://zenodo.org/records/5115136) for validating the automated classification system prototype.

---

### Step 2: Data Pre-processing
Once the datasets are acquired, use the provided notebooks to prepare the data for training:

* **Natural Events**: Run `EQ_EP_SS_data_set_processing.ipynb` to process Earthquake, Explosion, and Collapse data from DiTing 2.0.  
* **Noise Samples**: Run `noise_dataset_process.ipynb` to prepare noise data.

---

### Step 3: Running Benchmarking Experiments
Navigate to the `DiTing2.0_benchmark_runner/` directory. You can run individual tasks or execute concurrent experiments for model evaluation.

#### A. Single Task Execution (单任务运行)
Run the classification tasks on a single GPU (3-class or 4-class modes).

```bash
# Run 3-class classification (EQ, EP, SS)
CUDA_VISIBLE_DEVICES=0 python run_task.py --mode 3class --device cuda:0

# Run 4-class classification (including Expert_A/B/C and Junior_A noise subsets)
CUDA_VISIBLE_DEVICES=0 python run_task.py --mode 4class --device cuda:0
```
#### B. Concurrent Multi-GPU Execution (多 GPU 并行运行)
To maximize efficiency, you can run both tasks simultaneously on two separate GPUs:
```bash
# GPU 0 will handle 3-class / GPU 1 will handle 4-class subsets
bash run_dual_gpu.sh
```
Note: Execution logs will be automatically written to logs_3class.txt and logs_4class.txt for real-time monitoring.

### Step 4: Prototype System Execution
To launch the automated system prototype with the Graphical User Interface (GUI):
```bash
cd "Seismic Event Classification System"
python main.py
```
## 📊 Outputs & Visualization
### 1. Experimental Performance Benchmarks
The table below details the performance (Accuracy and F1-score) of various architectures under different input features and augmentation strategies.

| Input | Models | No Data Augmentation (Acc / F1) | Data Augmentation (Acc / F1) | Voting# |
| :--- | :--- | :--- | :--- | :--- |
| **STFT** | AlexNet | 82.36 / 82.27 | 89.94 / 89.94 | -- |
| | GoogleNet | 85.02 / 85.07 | **91.53 / 91.55** | -- |
| | ResNet | 83.86 / 83.86 | 91.29 / 91.30 | -- |
| | VGG | **86.83 / 86.80** | 89.99 / 90.15 | -- |
| | ViT | 77.76 / 77.79 | 81.86 / 81.92 | -- |
| | CCT | 76.19 / 76.43 | 85.98 / 86.00 | -- |
| **MFCC** | AlexNet | 86.88 / 86.91 | 89.94 / 89.94 | **98.01** |
| | GoogleNet | 87.22 / 87.18 | 89.81 / 89.85 | 97.69 |
| | ResNet | 86.23 / 86.23 | **90.57 / 90.62** | **98.01** |
| | VGG | **87.58 / 87.58** | 90.25 / 90.33 | 96.01 |
| | ViT | 86.44 / 86.41 | 89.68 / 89.69 | 97.06 |
| | CCT | 84.92 / 84.92 | 86.88 / 87.40 | 96.47 |
| **MFCC (CapsNet)** | CapsNet | 87.53 / 87.52 | 90.55 / 90.56 | 97.13 |
| | CapsNet+SE | 80.53 / 80.38 | 90.17 / 90.20 | 96.54 |
| | CapsNet+Res | **88.07 / 88.11** | **91.08 / 91.10** | 97.52 |
| | CapsNet+Res+SE| 87.50 / 87.57 | 90.76 / 90.76 | 97.87 |
| | CapsNet+CCT | 87.32 / 87.37 | 89.73 / 89.74 | 97.06 |

> **Notes**:
> 1. **Voting#** indicates performance after multi-station voting.
> 2. The combination of **MFCC + ResNet/AlexNet** with multi-station voting achieved the highest identification accuracy of **98.01%**.
> 3. CapsNet variants show robust performance, particularly with Residual connections (**CapsNet+Res**).

### 2. Automated Classification System Interface
The functional prototype of the **Seismic Event Classification System** provides an intuitive Graphical User Interface (GUI) for end-to-end seismic analysis, including waveform loading, STA/LTA detection, and deep learning-based classification.

![System GUI Interface](Interface_Image_%20Example.jpg)

*Figure: Graphical User Interface demonstrating the integrated seismic analysis and event classification pipeline.*


## 📝 Citation
1. Research Paper (In Revision)
* Zhenyu Pei, Xinyu Yang, Shiyu Liang, Zeyuan Zhong, Wenjing Xu, Lihua Fang, Kai Deng, Jun Hu, and Ke Jia. CapsNet-Enhanced Seismic Event Classification: Benchmarking and System Deployment on the DiTing 2.0 Dataset. Geophysical Journal International (In Revision, 2026). 

2. Dataset Citation
If you use the DiTing 2.0 dataset in your work, please cite it according to the official requirements:
* Zhao, M., Xiao, Z.W., Chen, S., Zhang, B., et al., 2023. Diting Dataset 2.0 – multi-functional large-scale artificial intelligence training data set from Chinese Seismic Network. [EB/OL]. https://data.earthquake.cn, 2023. DOI: 10.12080/nedc.11.ds.2023.0002.

## 📜 License

This project is licensed under the **MIT License**. You are free to use, modify, and distribute the code for research and academic purposes, provided that proper credit is given to the original authors and data providers. See the [LICENSE](LICENSE) file for more details.

---
**Maintained by**: [Kaixinzhenyu](https://github.com/Kaixinzhenyu)
