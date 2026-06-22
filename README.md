# Adaptive Inference Engine for Resource-Constrained Wearable Devices

## Overview

This project implements an **Adaptive Inference Engine** for Human Activity Recognition (HAR) on resource-constrained wearable devices. The system dynamically selects between lightweight and complex machine learning models based on prediction confidence, optimizing for both accuracy and computational efficiency.

### Key Features

- **Dual-Model Architecture**: Lightweight fast model + complex accurate model
- **Adaptive Gating Mechanism**: Intelligently switches between models based on confidence
- **Resource Optimization**: Minimizes computational overhead on edge devices
- **Activity Classification**: Recognizes activities like walking, sitting, standing, etc.
- **Performance Analysis**: Comprehensive evaluation metrics and comparisons

### Dataset

- **UCI HAR (Human Activity Recognition) Dataset**
- 7,352 training samples and 2,947 test samples
- 561 extracted features from smartphone sensors (accelerometer, gyroscope)
- 6 activity classes: WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING

---

## Project Structure

```
📦 Adaptive-Inference-Engine-for-Resource-Constrained-Wearable-Devices
├── 01_data_collection_eda.ipynb          # Data loading, preprocessing & EDA
├── 02_baseline_model.ipynb               # Training baseline ML models
├── 03_complexity_confidence.ipynb        # Model complexity analysis & confidence
├── 04_gating_mechanism.ipynb             # Adaptive gating implementation
├── 05_evaluation_analysis.ipynb          # Comprehensive evaluation & results
└── README.md                              # This file
```

### Notebook Descriptions

| Notebook | Purpose |
|----------|---------|
| **01_data_collection_eda.ipynb** | Download UCI HAR dataset, data preprocessing, exploratory data analysis, and feature engineering |
| **02_baseline_model.ipynb** | Train baseline models (Random Forest, SVM, Logistic Regression) for comparison |
| **03_complexity_confidence.ipynb** | Analyze model complexity, measure prediction confidence, and identify confidence thresholds |
| **04_gating_mechanism.ipynb** | Implement the adaptive gating mechanism that routes inputs to lightweight or complex models |
| **05_evaluation_analysis.ipynb** | Evaluate performance metrics, compare models, and analyze resource efficiency gains |

---

## Prerequisites

### Required Libraries

```
numpy >= 1.21.0
pandas >= 1.3.0
scikit-learn >= 0.24.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
jupyter >= 1.0.0
```

### Hardware Requirements

- Minimum: 4GB RAM
- Recommended: 8GB RAM for smooth execution
- GPU optional but not required

---

## Installation & Setup

### Option 1: Local Machine

#### 1. Clone the Repository
```bash
git clone https://github.com/araj1010/Adaptive-Inference-Engine-for-Resource-Constrained-Wearable-Devices.git
cd Adaptive-Inference-Engine-for-Resource-Constrained-Wearable-Devices
```

#### 2. Create Virtual Environment (Recommended)
```bash
# Using venv
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Using conda
conda create -n har-adaptive python=3.9
conda activate har-adaptive
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

#### 4. Start Jupyter Notebook
```bash
jupyter notebook
```

### Option 2: Google Colab (Recommended for Beginners)

1. Open [Google Colab](https://colab.research.google.com/)
2. Click `File` → `Open notebook` → `GitHub`
3. Enter the repository URL: `https://github.com/araj1010/Adaptive-Inference-Engine-for-Resource-Constrained-Wearable-Devices`
4. Select any notebook to open
5. Run cells directly in the browser (GPU available for free)

---

## Running the Project

### Sequential Execution (Recommended)

Execute the notebooks in order for optimal understanding and reproducibility:

#### Step 1: Data Collection & EDA
```
Run: 01_data_collection_eda.ipynb
Expected Output:
  - Train shape: (7352, 561)
  - Test shape: (2947, 561)
  - EDA visualizations
  - Clean preprocessed data
Time: ~5-10 minutes
```

#### Step 2: Baseline Model Training
```
Run: 02_baseline_model.ipynb
Expected Output:
  - Trained baseline models
  - Accuracy comparisons
  - Feature importance analysis
  - Model selection for lightweight/complex variants
Time: ~10-15 minutes
```

#### Step 3: Complexity & Confidence Analysis
```
Run: 03_complexity_confidence.ipynb
Expected Output:
  - Model complexity metrics (parameters, latency)
  - Prediction confidence distributions
  - Optimal confidence thresholds
  - Confusion matrices
Time: ~8-12 minutes
```

#### Step 4: Adaptive Gating Implementation
```
Run: 04_gating_mechanism.ipynb
Expected Output:
  - Gating mechanism implementation
  - Model routing decision process
  - Inference latency comparison
  - Energy efficiency estimation
Time: ~10-15 minutes
```

#### Step 5: Evaluation & Analysis
```
Run: 05_evaluation_analysis.ipynb
Expected Output:
  - Comprehensive performance metrics
  - Accuracy vs. Efficiency trade-offs
  - Resource savings analysis
  - Final conclusions and insights
Time: ~5-10 minutes
```

---

## Execution Workflow

### Quick Start in Jupyter

1. **Open the first notebook:**
   ```bash
   jupyter notebook 01_data_collection_eda.ipynb
   ```

2. **Run all cells** (Kernel → Restart & Run All) or execute cells sequentially

3. **Move to next notebook** after verification

4. **Expected Total Runtime:** ~40-60 minutes for complete project

### Google Colab Quick Start

1. Open notebook in Colab
2. **Automatically enabled**: GPU acceleration (optional)
3. Execute cells top-to-bottom
4. No additional setup needed (Colab handles dependencies)

---

## Key Outputs & Results

### Data Insights
- Activity distribution and class balance
- Sensor data patterns and correlations
- Feature scaling effects

### Model Performance
- Baseline accuracy: ~95%+ (Random Forest)
- Lightweight model accuracy: ~92%+
- Complex model accuracy: ~96%+

### Adaptive Engine Benefits
- **30-50%** reduction in average inference time
- **40-60%** reduction in computational load
- **>95%** accuracy maintained with adaptive gating

### Resource Efficiency
- Memory usage comparison (lightweight vs. complex models)
- Latency analysis across different thresholds
- Energy efficiency gains on wearable devices

---

## Configuration & Customization

### Adjust Confidence Threshold
In **04_gating_mechanism.ipynb**:
```python
CONFIDENCE_THRESHOLD = 0.85  # Adjust between 0.0 and 1.0
# Lower = more complex model usage (higher accuracy)
# Higher = more lightweight model usage (lower latency)
```

### Change Activity Classes
In **01_data_collection_eda.ipynb**:
```python
# Modify activity_map to include/exclude specific activities
activity_map = {
    1: 'WALKING',
    2: 'WALKING_UPSTAIRS',
    3: 'WALKING_DOWNSTAIRS',
    4: 'SITTING',
    5: 'STANDING',
    6: 'LAYING'
}
```

### Modify Model Parameters
Each notebook includes hyperparameter cells:
```python
# Example: Random Forest
rf_model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **Dataset download fails** | Check internet connection; manually download from [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) |
| **Memory errors** | Reduce batch size or filter data; use Colab GPU |
| **Import errors** | Run `pip install --upgrade scikit-learn pandas numpy` |
| **Slow execution** | Enable GPU in Colab; reduce dataset size for testing |
| **Missing files** | Ensure all .ipynb files are in the same directory |

---

## Performance Benchmarks

### Environment
- CPU: Intel Core i7 / Google Colab CPU
- RAM: 8GB+
- Dataset: UCI HAR (10,299 samples)

### Timing Results
| Stage | Time (Local) | Time (Colab) |
|-------|-------------|-------------|
| Data Loading & EDA | 5-10 min | 3-5 min |
| Model Training | 10-15 min | 8-12 min |
| Confidence Analysis | 8-12 min | 5-10 min |
| Gating Mechanism | 10-15 min | 8-10 min |
| Evaluation | 5-10 min | 3-5 min |
| **Total** | **40-60 min** | **30-45 min** |

---

## Project Methodology

### Architecture
1. **Lightweight Model**: Logistic Regression / Simple Decision Tree
   - Low latency, minimal resources
   - Suitable for edge devices

2. **Complex Model**: Random Forest / SVM
   - High accuracy, moderate resources
   - Used for uncertain predictions

3. **Gating Network**: Confidence-based router
   - Routes based on prediction confidence
   - Threshold-tunable

### Adaptive Logic
```
Input → Feature Vector
        ↓
    Lightweight Model
        ↓
    Confidence ≥ Threshold?
        ├─ Yes → Output prediction (lightweight)
        └─ No → Complex Model → Output prediction
```

---

## Expected Learning Outcomes

- **Edge Computing**: Deploying ML on resource-constrained devices
- **Model Optimization**: Trading off accuracy for efficiency
- **HAR Systems**: Building activity recognition systems
- **Adaptive Systems**: Dynamic model selection strategies
- **Performance Analysis**: Comprehensive ML evaluation metrics

---

## License

This project is provided as-is for educational and research purposes.

---

## Contact & Support

**Author**: Anand Raj (@araj1010)  
**GitHub**: [araj1010](https://github.com/araj1010)

For questions, issues, or contributions:
- Open an issue on GitHub
- Check existing documentation
- Refer to UCI HAR dataset documentation

---

## References

- [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
- [Edge Computing & ML](https://arxiv.org/abs/2009.08147)
- [Adaptive Inference Engines](https://arxiv.org/abs/1905.12949)
- Scikit-learn Documentation

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | June 2026 | Initial release with 5 notebooks |

---

**Last Updated**: June 22, 2026  
**Status**: Active & Ready to Use ✅
