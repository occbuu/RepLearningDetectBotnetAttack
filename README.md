# Botnet Detection Project

A comprehensive machine learning project for detecting botnets using deep neural networks and traffic analysis. This project implements multiple models with different configurations to achieve robust botnet detection and classification.

## Project Overview

This project focuses on identifying and classifying botnet traffic using advanced machine learning techniques. It includes three different model configurations with varying parameters to test and optimize detection accuracy.

## Project Structure

```
├── 32botnet_detection_colab.ipynb       # Model with 32-unit configurations
├── 64botnet_detection_colab.ipynb       # Model with 64-unit configurations
├── 128botnet_detection_colab copy.ipynb # Model with 128-unit configurations
└── README.md                             # Project documentation
```

## Models

The project contains three Jupyter notebooks, each implementing a botnet detection model with different neural network configurations:

### 1. **32botnet_detection_colab.ipynb**
- Uses 32-unit dense layers
- Lightweight configuration for resource-constrained environments
- Suitable for rapid prototyping and initial experiments

### 2. **64botnet_detection_colab.ipynb**
- Uses 64-unit dense layers
- Balanced between model complexity and computational resources
- Recommended for production environments

### 3. **128botnet_detection_colab copy.ipynb**
- Uses 128-unit dense layers
- More complex model architecture
- Better for achieving higher accuracy on complex datasets

## Features

- **Data Preprocessing**: Automatic handling of network traffic data and feature extraction
- **Model Training**: Multi-epoch training with validation and loss tracking
- **Visualization**: Loss curves and performance metrics plotting
- **Colab Integration**: Automatic Google Drive mounting and environment detection
- **Flexible Configuration**: Works on both Google Colab and local machines

## Technical Stack

- **Python 3.x**
- **TensorFlow/Keras** - Deep learning framework
- **NumPy** - Numerical computing
- **Matplotlib & Seaborn** - Data visualization
- **Scikit-learn** - Machine learning utilities

## Setup & Usage

### Running on Google Colab

1. Upload the notebook to Google Colab
2. The notebook will automatically detect the Colab environment
3. Grant access to Google Drive for data storage
4. Run all cells sequentially

### Running Locally

1. Ensure Python 3.x is installed with required dependencies:
   ```bash
   pip install tensorflow numpy matplotlib seaborn scikit-learn
   ```
2. Install Jupyter Notebook or JupyterLab:
   ```bash
   pip install jupyter
   ```
3. Launch the notebook:
   ```bash
   jupyter notebook 32botnet_detection_colab.ipynb
   ```

## Usage Examples

Each notebook follows this workflow:

1. **Data Preparation** - Load and preprocess network traffic data
2. **Model Definition** - Define neural network architecture
3. **Training** - Train the model with validation monitoring
4. **Evaluation** - Assess model performance
5. **Visualization** - Generate loss curves and performance plots
6. **Export** - Save visualizations and trained models

## Model Architecture

All models follow a similar architecture with varying layer sizes:

```
Input Layer
    ↓
Dense Layer (32/64/128 units, ReLU activation)
    ↓
Dropout Layer
    ↓
Dense Layer (32/64/128 units, ReLU activation)
    ↓
Dropout Layer
    ↓
Output Layer (Sigmoid activation for binary classification)
```

## Performance Metrics

The models are evaluated using:
- **Training Loss** - Cross-entropy loss during training
- **Validation Loss** - Cross-entropy loss on validation set
- **Accuracy** - Classification accuracy
- **Precision, Recall, F1-Score** - Detailed performance metrics

## Key Parameters

- **Batch Size**: Configurable for memory optimization
- **Epochs**: 12 epochs as default (adjustable)
- **Learning Rate**: Automatically optimized by Adam optimizer
- **Dropout Rate**: 0.2-0.3 for regularization

## Output Files

The notebooks generate:
- `loss_plot_12_epochs.pdf` - High-quality loss visualization (vector format)
- `loss_plot_12_epochs.png` - PNG version of loss plot (300 DPI)
- Trained model files (`.h5` or `.keras` format)
- Classification reports and confusion matrices

## Configuration

The notebooks include automatic environment detection:

```python
# Auto-detect Colab environment
try:
    import google.colab
    IS_COLAB = True
except ImportError:
    IS_COLAB = False
```

You can manually override this by setting `IS_COLAB = True` or `IS_COLAB = False` as needed.

## Data Requirements

- Network traffic dataset in compatible format (CSV, HDF5, or NumPy)
- Features should include network flow characteristics
- Labels should indicate botnet vs. benign traffic
- Minimum dataset size: ~5000 samples for reliable training

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Google Drive not mounting | Check Colab authentication and Drive permissions |
| CUDA out of memory | Reduce batch size or use smaller model (32-unit) |
| Data not found | Verify data path and file permissions |
| Training too slow | Enable GPU acceleration in Colab settings |

## Results Interpretation

- **Training Loss**: Should decrease steadily over epochs
- **Validation Loss**: Should decrease and stabilize (overfitting if diverges from training loss)
- **Gap between train/val loss**: Indicates model generalization quality

Example from the project:
```
Epoch 1:  Training Loss = 0.0606, Validation Loss = 0.5069
Epoch 12: Training Loss = 0.0000, Validation Loss = 0.0028
```

## Future Improvements

- [ ] Add ensemble methods combining all three models
- [ ] Implement real-time detection capabilities
- [ ] Add explainability features (SHAP, attention mechanisms)
- [ ] Support for multi-class botnet classification
- [ ] Performance optimization for edge devices
- [ ] Integration with network monitoring tools

## References

- Network traffic analysis literature
- Deep learning best practices for anomaly detection
- TensorFlow/Keras documentation

## License

This project is part of the Paper2026 research initiative.

## Contact & Support

For questions or issues related to this project, please refer to the project documentation or contact the development team.

---

**Last Updated**: February 2026  
**Project Status**: Active Development
