# SVM Parameter Optimization Assignment

This repository contains a Jupyter Notebook that performs parameter optimization for Support Vector Machines (SVM) using the `NuSVR` model from `scikit-learn`. The dataset used is the Wine dataset, which contains 3 classes and 13 features.

## Objectives

- Split the dataset into 10 different samples (each split is 70% training, 30% testing).
- For each sample:
  - Perform 100 iterations of random hyperparameter search.
  - Hyperparameters include: `kernel`, `nu`, and `epsilon`.
- Store the best parameters and corresponding accuracy for each sample.
- Save results to a CSV file.
- Plot the convergence graph for the sample with the highest accuracy.

## Files

- `SVM_Parameter_Optimization_Assignment.ipynb`: Main Jupyter notebook containing the analysis.
- `results_table.csv`: CSV file storing the best results for each sample.
- `convergence_graph.png`: Graph showing the convergence of accuracy across 100 iterations for the best performing sample.

## Requirements

- Python 3.x
- scikit-learn
- pandas
- matplotlib
- numpy

## Usage

1. Open the notebook using Jupyter:
   ```bash
   jupyter notebook SVM_Parameter_Optimization_Assignment.ipynb
