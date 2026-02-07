Application of Representation Learning in Detecting Botnet Attacks

Author: Hieu Le Ngoc

Affiliation: Faculty of Information Technology, Van Hien University

Status: Under Revision at Scientific Reports

Abstract

Botnet detection remains a perennial and critical challenge in cybersecurity. This paper introduces a robust framework that significantly enhances botnet detection by combining advanced feature engineering—such as octet splitting for IP addresses—with a sophisticated representation learning technique using the Hilbert space-filling curve to transform network flows into 2D images.

We address class imbalance using SMOTE and Focal Loss. Evaluating strictly on a cross-scenario basis (Training on CTU-13 Scenario 8, Testing on unseen Scenario 10), our ResNet-18 based model achieved 98.34% accuracy and a weighted F1-score of 98.38%, outperforming traditional Random Forest baselines which failed to generalize.

Methodology

The Pipeline

- Preprocessing: Raw NetFlows are cleaned. Background and Normal traffic are merged into a "Benign" class.

Feature Engineering:
- IP Octet Splitting (Source/Dest IP -> 4 features each).
- Frequency Encoding for Ports/Protocols.
- Min-Max Scaling.

Final Vector Size: N = 19.

Hilbert Transformation: The 1D feature vector is mapped to a 2D grid (32×32, 64×64, etc.) using the Hilbert Space-Filling Curve to preserve data locality.

Classification: A modified ResNet-18 (1-channel input) trained with Focal Loss to handle class imbalance.

Getting Started

Prerequisites
The code is implemented in Python 3.8+ using PyTorch.

Clone the repo:

```bash
git clone https://github.com/occbuu/RepLearningDetectBotnetAttack.git
cd RepLearningDetectBotnetAttack
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Dataset
This project uses the CTU-13 Dataset. Due to size constraints, the raw PCAP/NetFlow files are not included in this repo.

Training: Scenario 8 (Murlo)
Testing: Scenario 10 (Rbot)

Download the dataset from the Stratosphere IPS Repository.

Running the Model

To train the model on Scenario 8 and evaluate on Scenario 10:

```bash
python main.py --mode train --image_size 32 --epochs 12
```

To run inference on a new pcap file:

```bash
python main.py --mode predict --input data/my_traffic.csv
```

Results

Model Configuration	Accuracy	F1-Score (Weighted)	Recall (Botnet)
ResNet-18 (32x32)	98.34%	98.38%	98.34%
ResNet-18 (64x64)	98.07%	98.07%	98.07%
ResNet-18 (128x128)	97.12%	97.12%	97.12%
Random Forest	91.88%	90.99%	0.02% (Fail)

Repository Structure

- src/hilbert_curve.py: Implementation of the algorithm to map 1D arrays to 2D Hilbert images.
- src/model.py: Custom ResNet-18 architecture modified for grayscale (1-channel) input.
- src/utils.py: Implementation of Focal Loss and metric calculations.

Citation

Le Ngoc, H. (2026). Application of Representation Learning in Detecting Botnet Attacks. Scientific Reports (Under Review).

Final Checklist for the Editor

- Run your code on a clean computer (or Google Colab) to make sure requirements.txt actually works.
- Upload these changes to GitHub.
- Check the link: Ensure https://github.com/occbuu/RepLearningDetectBotnetAttack is public and accessible.
 