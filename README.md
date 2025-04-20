# 📌 SVM Parameter Optimization

This project aims to optimize the performance of a Support Vector Machine (SVM) model by tuning hyperparameters on a multi-class classification problem using the Wine dataset. We use `NuSVR` from scikit-learn, although the task is classification — to simulate a regression-to-classification approximation scenario and explore parameter effects.

---

## 🔍 Methodology

1. **Dataset**:  
   We use the Wine dataset from the UCI ML repository, which has:
   - 178 samples
   - 13 features
   - 3 output classes

2. **Sampling**:
   - The dataset is split into 10 different **random samples**.
   - Each sample uses a 70% training and 30% testing split with different `random_state` values (from 0 to 9).

3. **Hyperparameter Optimization**:
   - For each sample, we perform **100 iterations** of randomized hyperparameter tuning.
   - The hyperparameters tuned are:
     - `kernel`: one of `'rbf'`, `'poly'`, `'sigmoid'`
     - `nu`: range from 0.1 to 0.9
     - `epsilon`: range from 0.01 to 1.0
   - Each combination is tested using `NuSVR`, and prediction accuracy is calculated by rounding predicted values and comparing with true labels.

4. **Metrics**:
   - For every sample, we record:
     - The best accuracy out of 100 trials
     - The corresponding set of hyperparameters

5. **Visualization**:
   - For the sample with the highest accuracy, a **convergence graph** is plotted to show how the accuracy changed over 100 iterations.

---

## 📊 Result Table (Sample)

| Sample | Best Accuracy (%) | Best Params                          |
|--------|-------------------|--------------------------------------|
| S1     | 92.31             | {'kernel': 'rbf', 'nu': 0.5, 'epsilon': 0.1} |
| S2     | 89.74             | {'kernel': 'poly', 'nu': 0.3, 'epsilon': 0.2} |
| ...    | ...               | ...                                  |
| S10    | 94.87             | {'kernel': 'sigmoid', 'nu': 0.7, 'epsilon': 0.05} |

> 🔸 The CSV file `results_table.csv` contains complete data for all 10 samples.

---

## 📈 Result Graph: Convergence Plot

The graph `convergence_graph.png` shows how the accuracy evolved over 100 iterations for the **sample with the highest accuracy**. It helps us observe:

- How quickly the model discovered good parameters
- Stability in accuracy over time
- Whether the optimization is still improving after 100 iterations

**Graph Details:**
- **X-axis**: Iteration number (1 to 100)
- **Y-axis**: Accuracy (%) on the test set
- **Line Plot**: Tracks how accuracy fluctuated with different parameter combinations

---

## 📂 Files

- `SVM_Parameter_Optimization_Assignment.ipynb`: Jupyter notebook with full code and outputs
- `results_table.csv`: Best results from all 10 samples
- `convergence_graph.png`: Visualization of accuracy vs. iteration for the top sample

---

## 📦 Dependencies

- Python 3.x
- scikit-learn
- pandas
- numpy
- matplotlib

Install them via:

```bash
pip install scikit-learn pandas numpy matplotlib
