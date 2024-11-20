# Neural Network Challenge 2  

## Description  
This project implements a multi-task neural network model to predict employee **Attrition** (Yes/No) and their **Department** (Human Resources, Research & Development, Sales). The model incorporates advanced techniques like shared layers, layer normalization, dropout regularization to achieve robust predictions.  

## Table of Contents  
1. [Usage](#usage)  
2. [Model Development](#model-development)  
3. [Results](#results)  
4. [Next Steps](#next-steps)  
5. [Credits](#credits)  

## Usage  
To replicate or adapt this project:  
1. Clone the repository.  
2. Run the Jupyter notebook or Python scripts to train the model.  
3. Evaluate the model using the test set provided.  

## Model Development  

### Feature Engineering and Inspection  
- **Data Encoding:**  
   - Categorical variables were one-hot encoded, and numerical variables were scaled using `StandardScaler`.  
   - Only features with high correlation or importance to the target were retained to minimize noise.  

---

### Shared Layers  
The shared layers form the backbone of the model, allowing both tasks (Attrition and Department prediction) to leverage common patterns:  
- **Dense Layers:** Fully connected layers with increasing capacity for learning complex patterns.  
- **Batch Normalization:** Normalizes activations to stabilize and accelerate training.  
- **ReLU Activation:** Adds non-linearity, essential for learning relationships in data.  
- **Dropout:** A rate of 10%-20% was applied to prevent overfitting while maintaining model capacity.  

---

### Branch-Specific Layers  
Separate layers for **Attrition** and **Department**:  
- **Attrition:** Binary classification with a `sigmoid` activation for a Yes/No output.  
- **Department:** Multiclass classification with a `softmax` activation to output probabilities for three categories.  

## Results  

### **Accuracy and Loss Metrics**  
- **Attrition Task:**  
  - Accuracy: **88.04%**  
  - Loss: **0.324**  

- **Department Task:**  
  - Accuracy: **97.28%**  
  - Loss: **0.086**  

- **Overall Loss:** **0.412**  

---

### **Confusion Matrices**  

#### **_Attrition_**  
|                     | Predicted No (0) | Predicted Yes (1) |  
|---------------------|------------------|-------------------|  
| **Actual No (0)**   | 298              | 15                |  
| **Actual Yes (1)**  | 29               | 26                |  

#### **_Department_**  
|                         | Predicted Human Resources (0) | Predicted R&D (1) | Predicted Sales (2) |  
|-------------------------|------------------------------|-------------------|---------------------|  
| **Actual HR (0)**       | 8                            | 2                 | 0                   |  
| **Actual R&D (1)**      | 1                            | 239               | 3                   |  
| **Actual Sales (2)**    | 1                            | 3                 | 111                 |  

---

### **Classification Reports**  

#### **_Attrition_**  
| Metric          | Precision | Recall | F1-Score | Support |  
|-----------------|-----------|--------|----------|---------|  
| **No (0)**      | 0.91      | 0.95   | 0.93     | 313     |  
| **Yes (1)**     | 0.63      | 0.47   | 0.54     | 55      |  
| **Weighted Avg**| 0.87      | 0.88   | 0.87     | 368     |  

#### **_Department_**  
| Metric                | Precision | Recall | F1-Score | Support |  
|-----------------------|-----------|--------|----------|---------|  
| **Human Resources**   | 0.80      | 0.80   | 0.80     | 10      |  
| **Research & Dev.**   | 0.98      | 0.98   | 0.98     | 243     |  
| **Sales**             | 0.97      | 0.97   | 0.97     | 115     |  
| **Weighted Avg**      | 0.97      | 0.97   | 0.97     | 368     |  

---

### **Accuracy and Loss per Epoch**  

![Accuracy and Loss](/acc_loss_plot.png)

---

## Next Steps  
- **Feature Engineering:** Apply Principal Component Analysis (PCA) to evaluate dimensionality reduction.  
- **Sampling:** Experiment with Synthetic Minority Oversampling Technique (SMOTE) for further balancing.  
- **Model Enhancements:** Explore alternative architectures such as transformer-based models for sequence tasks.  

---

## Credits  
Starter code for this assignment was provided by [edX bootcamp](https://www.edx.org/boot-camps).

The `utils` folder includes helpful code from a previous [project](https://github.com/Corey-Holton/Group_project_2) of mine that enabled:
- **VIF calculations** to handle multicollinearity effectively
- **Highlighted graphs and correlation tables** for data visualization
