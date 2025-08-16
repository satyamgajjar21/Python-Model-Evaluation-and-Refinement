# Model Evaluation and Refinement

This repository contains a Jupyter Notebook/Lab that walks through **model evaluation and refinement techniques** in machine learning, using Python and Scikit-learn.  

The lab is designed as part of the IBM Data Science/AI curriculum and focuses on evaluating predictive models, handling overfitting, applying regularization, and using grid search for hyperparameter tuning.  

---

## Objectives

By the end of this lab, you will be able to:

- Split data into training and testing sets  
- Evaluate models using **R²** and cross-validation  
- Understand **overfitting vs underfitting** and how to handle them  
- Apply **Polynomial Regression** and visualize results  
- Use **Ridge Regression** for regularization  
- Perform **Grid Search** to optimize hyperparameters  

---

## Table of Contents

1. [Model Evaluation](#model-evaluation)  
2. [Over-fitting, Under-fitting and Model Selection](#overfitting-underfitting-and-model-selection)  
3. [Ridge Regression](#ridge-regression)  
4. [Grid Search](#grid-search)  

---

## Dataset

The dataset used is `module_5_auto.csv`, hosted on IBM Cloud Object Storage.  
It contains car attributes (e.g., horsepower, engine size, mpg) and price values used to build regression models.

📥 [Download Dataset](https://cocl.us/DA101EN_object_storage?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDA0101ENSkillsNetwork20235326-2021-01-01)

---

## Running the Lab

### Option 1: Skills Network / JupyterLite
If running in the **browser environment** (Skills Network or JupyterLite), install dependencies with `piplite`:

```python
import piplite
await piplite.install(['pandas', 'matplotlib', 'scipy', 'scikit-learn', 'seaborn'])
