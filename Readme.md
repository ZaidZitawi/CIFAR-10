# CNN Classification on CIFAR-10 Subset

This project implements and compares two Convolutional Neural Network (CNN) architectures for image classification on a 4-class subset of the CIFAR-10 dataset. The models are evaluated using various metrics and visualizations.

## Features

- Two CNN architectures:
  - **BaselineCNN**: Simple architecture with 1 convolutional layer
  - **SimpleCNN**: Advanced architecture with multiple layers and regularization
- Training/evaluation pipeline with metrics:
  - Accuracy, Precision, Recall, F1-score
  - Confusion matrices
  - Training curves (loss & accuracy)
- Detailed performance comparison and insights

## Requirements

- Python 3.6+
- PyTorch 1.8+
- torchvision 0.9+
- scikit-learn 0.24+
- matplotlib 3.4+
- seaborn 0.11+



# CNN Classification on CIFAR-10 Subset

## Results

### Training Performance

**Baseline CNN Training (10 epochs):**
**Advanced CNN Training (10 epochs):**

*Total training time: ~12 minutes*

### Performance Metrics

| Metric          | Baseline CNN | Advanced CNN |
|-----------------|--------------|--------------|
| Accuracy        | 66%          | 87%          |
| Precision       | 66%          | 87.5%        |
| Recall          | 66%          | 87.5%        |
| F1-Score        | 65%          | 87.5%        |

*Note: Metrics may vary slightly between builds due to random initialization*

### Training Progression

**Baseline CNN:**
| Epoch | Train Loss | Train Acc | Test Loss | Test Acc |
|-------|------------|-----------|-----------|----------|
| 1     | 1.1179     | 51.96%    | 1.0193    | 58.35%   |
| 5     | 0.8832     | 64.21%    | 0.8801    | 63.88%   |
| 10    | 0.8062     | 67.42%    | 0.8368    | 65.97%   |

**Advanced CNN:**
| Epoch | Train Loss | Train Acc | Test Loss | Test Acc |
|-------|------------|-----------|-----------|----------|
| 1     | 0.9085     | 61.39%    | 0.6480    | 74.12%   |
| 5     | 0.4344     | 83.96%    | 0.5044    | 80.97%   |
| 10    | 0.3244     | 88.00%    | 0.3481    | 87.48%   |

### Visualizations

1. **Confusion Matrices:**
   - Baseline CNN: Shows predominant confusion between similar classes
   - Advanced CNN: Demonstrates clearer diagonal dominance (better class separation)

2. **Training Curves:**
   - Baseline Model: Shows gradual improvement with some oscillation in test accuracy
   - Advanced Model: Rapid early improvement followed by stable convergence

### Comparative Analysis

| Aspect          | Baseline CNN         | Advanced CNN          |
|-----------------|----------------------|-----------------------|
| Peak Accuracy   | 66%                 | 87.5%                |
| Learning Speed  | Slow convergence    | Rapid early progress |
| Overfitting     | Moderate            | Well-controlled      |
| Class Separation| Frequent confusion  | Clear distinction    |
| Training Stability| Oscillating metrics| Stable convergence   |

**Key Observations:**
1. The Advanced CNN achieves 32% relative accuracy improvement over Baseline
2. Final Advanced CNN test accuracy (87.5%) nearly matches training accuracy (88%)
3. Advanced model maintains high precision-recall balance (F1=87.5%)
4. Training curves show Advanced CNN reaches peak performance by epoch 7

## Discussion

### Training Insights
- **Advanced CNN** shows rapid early learning (74% accuracy in first epoch)
- Both models benefit from Adam optimizer's adaptive learning rates
- Advanced model's dropout (0.5) effectively prevents overfitting despite deeper architecture

### Error Analysis
- Baseline CNN confusion matrix shows systematic misclassifications
- Advanced CNN errors are more scattered, suggesting harder edge cases
- Both models show highest confusion between vehicle classes (truck/automobile)

### Performance Variance
- Metric fluctuations between builds (<±2%) due to:
  - Random weight initialization
  - Data loader shuffling
  - Non-deterministic GPU computations
- Consistent >85% accuracy for Advanced CNN across builds demonstrates robustness

## Visualizations (Generated During Execution)

1. **Confusion Matrices:**
   - Saved as `confusion_baseline.png` and `confusion_advanced.png`
   
2. **Training Curves:**
   - Loss progression: `training_loss.png`
   - Accuracy development: `training_accuracy.png`

*Total training time: ~12 minutes*

### Performance Metrics

| Metric          | Baseline CNN | Advanced CNN |
|-----------------|--------------|--------------|
| Accuracy        | 66%          | 87%          |
| Precision       | 66%          | 87.5%        |
| Recall          | 66%          | 87.5%        |
| F1-Score        | 65%          | 87.5%        |

*Note: Metrics may vary slightly between builds due to random initialization*

### Training Progression

**Baseline CNN:**
| Epoch | Train Loss | Train Acc | Test Loss | Test Acc |
|-------|------------|-----------|-----------|----------|
| 1     | 1.1179     | 51.96%    | 1.0193    | 58.35%   |
| 5     | 0.8832     | 64.21%    | 0.8801    | 63.88%   |
| 10    | 0.8062     | 67.42%    | 0.8368    | 65.97%   |

**Advanced CNN:**
| Epoch | Train Loss | Train Acc | Test Loss | Test Acc |
|-------|------------|-----------|-----------|----------|
| 1     | 0.9085     | 61.39%    | 0.6480    | 74.12%   |
| 5     | 0.4344     | 83.96%    | 0.5044    | 80.97%   |
| 10    | 0.3244     | 88.00%    | 0.3481    | 87.48%   |

### Visualizations

1. **Confusion Matrices:**
   - Baseline CNN: Shows predominant confusion between similar classes
   - Advanced CNN: Demonstrates clearer diagonal dominance (better class separation)

2. **Training Curves:**
   - Baseline Model: Shows gradual improvement with some oscillation in test accuracy
   - Advanced Model: Rapid early improvement followed by stable convergence

### Comparative Analysis

| Aspect          | Baseline CNN         | Advanced CNN          |
|-----------------|----------------------|-----------------------|
| Peak Accuracy   | 66%                 | 87.5%                |
| Learning Speed  | Slow convergence    | Rapid early progress |
| Overfitting     | Moderate            | Well-controlled      |
| Class Separation| Frequent confusion  | Clear distinction    |
| Training Stability| Oscillating metrics| Stable convergence   |

**Key Observations:**
1. The Advanced CNN achieves 32% relative accuracy improvement over Baseline
2. Final Advanced CNN test accuracy (87.5%) nearly matches training accuracy (88%)
3. Advanced model maintains high precision-recall balance (F1=87.5%)
4. Training curves show Advanced CNN reaches peak performance by epoch 7

## Discussion

### Training Insights
- **Advanced CNN** shows rapid early learning (74% accuracy in first epoch)
- Both models benefit from Adam optimizer's adaptive learning rates
- Advanced model's dropout (0.5) effectively prevents overfitting despite deeper architecture

## Visualizations (Generated During Execution)

1. **Confusion Matrices:**
   - Saved as `confusion_baseline.png` and `confusion_advanced.png`

     ![image](https://github.com/user-attachments/assets/26fa4f25-3904-4780-bcf2-7107be59c3e0)

     ![image](https://github.com/user-attachments/assets/2fb47972-f5ce-41d9-8b9e-de50a51c6bea)

   
2. **Training Curves:**
   - Loss progression: `training_loss.png`
   - Accuracy development: `training_accuracy.png`
  ![image](https://github.com/user-attachments/assets/c48a146a-0f48-4b76-812e-6b7b1be4c1be)

