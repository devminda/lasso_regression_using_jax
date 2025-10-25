# 🔍 Lasso Regression with JAX

This project implements **Lasso regression** using [JAX](https://github.com/google/jax), with a focus on efficient hyperparameter tuning and model evaluation. It combines **random search**, **refined grid search**, and **cross-validation** to build a robust and generalizable model.

---

## 🚀 Features

- ✅ **L1-Regularized Linear Regression** (Lasso) using JAX for fast, differentiable computation  
- 🔄 **Two-Stage Hyperparameter Tuning**:
  - Stage 1: Random search over λ (regularization) and learning rate
  - Stage 2: Focused grid search around best candidates
- 📊 **Cross-Validation** to assess model generalization and detect overfitting
- 📈 **Mini-Batch Gradient Descent** with JIT compilation for performance
- 📁 **Prediction Output** on test data saved to `predictions.csv`

---

## 📂 Project Structure

```
.
├── train.csv               # Training dataset
├── test.csv                # Test dataset
├── predictions.csv         # Output predictions
├── import numpy as np.txt  # Main JAX implementation script
└── README.md               # Project documentation
```

---

## 🧪 How It Works

1. **Data Preprocessing**  
   - Normalizes features  
   - Adds bias term  
   - Splits into training and validation sets

2. **Model Training**  
   - Uses JAX's `grad` and `jit` for efficient optimization  
   - Applies L1 penalty to encourage sparsity

3. **Hyperparameter Tuning**  
   - Randomly samples combinations of λ and learning rate  
   - Refines search around best-performing pair

4. **Cross-Validation**  
   - Performs k-fold CV to evaluate model stability

5. **Prediction**  
   - Applies trained model to test data  
   - Outputs predictions to CSV

---

## 🛠️ Requirements

- Python 3.8+
- JAX
- NumPy
- pandas
- scikit-learn

Install dependencies:

```bash
pip install jax jaxlib numpy pandas scikit-learn
```

---

## 📈 Example Output

```bash
Best Hyperparameters from Combined Search:
Lambda: 0.001, Learning Rate: 0.01
Validation MSE: 0.0234

Cross-Validation Results:
Fold 1: MSE = 0.0241
Fold 2: MSE = 0.0228
...
```

---

## 📬 Contact

For questions or collaboration, feel free to reach out via GitHub Issues or pull requests.

---

Let me know if you'd like a badge section, usage examples, or a license block added!
