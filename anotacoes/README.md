## Decision trees classification

### Key Concepts:
1. Tree Structure: Nodes represent features, branches represent decision rules, leaf nodes represent class predictions.

2. Split Criteria:
- Gini Impurity: Measures node's impurity/randomness
- Information Gain/Entropy: Quantifies information reduction from splits
- Measures how well a feature separates different classes

3. Model Interpretation:
- Feature Importance: Shows which attributes most influence classification
- Decision paths are easily visualizable and interpretable

### Performance Metrics:
- Accuracy: Percentage of correct predictions
- Precision: Proportion of true positive predictions
- Recall: Proportion of actual positives correctly identified
- F1 Score: Harmonic mean of precision and recall
- Confusion Matrix: Detailed breakdown of predictions

### Key Evaluation Steps:
1. Split data into training/testing sets
2. Train the model
3. Validate using cross-validation
4. Analyze performance metrics
5. Prune tree to prevent overfitting

### Practical Considerations:
- Works well with both numerical and categorical data
- Sensitive to training data variations
- Consider ensemble methods like Random Forest for improved performance



## Decision Tree - regression

### Key Concepts:
- Decision trees for regression predict continuous numerical values by recursively splitting data into subsets
- The model creates a tree-like structure where each internal node represents a feature and a splitting condition
- Leaf nodes contain the predicted numerical value for that subset of data

### Measurement and Interpretation:
1. Performance Metrics:
- Mean Squared Error (MSE): Average squared difference between predicted and actual values
- Root Mean Squared Error (RMSE): Square root of MSE, in original data units
- Mean Absolute Error (MAE): Average absolute difference between predictions and actual values
- R-squared (R²): Proportion of variance explained by the model (0-1 scale)

2. Tree Interpretation:
- Feature importance: Shows which features most significantly influence predictions
- Splitting criteria typically use variance reduction (minimizing variance within child nodes)
- Tree depth controls model complexity (deeper trees can overfit)

### Model Building Steps:
- Split data into training and testing sets
- Select features
- Choose splitting algorithm (e.g., CART - Classification and Regression Trees)
- Prune tree to prevent overfitting
- Validate model performance using cross-validation

## Cost of complexity
Cost of complexity in decision tree models refers to a technique for preventing overfitting by penalizing model complexity. 

Key points:

1. Core Concept:
- Balances model's predictive accuracy with its structural complexity
- Adds a penalty term to the model's error based on tree size/depth
- Helps prevent overfitting by discouraging unnecessarily complex trees

2. Calculation Methods:
- Reduced Error Pruning
- Cost Complexity Pruning (Minimum Error Pruning)
- Uses alpha (complexity parameter) to control tree simplification

3. Complexity Measurement:
- Number of terminal nodes
- Tree depth
- Total number of branches/splits
- Penalizes additional complexity beyond optimal predictive performance

4. Trade-offs:
- Lower complexity: Simpler model, potentially less accurate
- Higher complexity: More detailed model, risk of overfitting
- Goal: Find optimal balance between model complexity and predictive power

5. Implementation:
- Systematically prune branches
- Use cross-validation to select optimal complexity parameter
- Minimize combined error (prediction error + complexity penalty)


### K-fold Cross-Validation Overview:

Purpose:
- Robust method for model performance evaluation
- Divides dataset into K equal-sized subsets (folds)
- Reduces bias and variance in model assessment

Process:
1. Split data into K equal groups
2. Iteratively use K-1 groups for training, 1 group for testing
3. Rotate testing group K times
4. Calculate average performance across all iterations

Key Benefits:
- More reliable performance estimate
- Helps prevent overfitting
- Works well with limited data
- Provides comprehensive model assessment

Common Configurations:
- 5-fold: Standard practice
- 10-fold: Most typical approach
- Leave-One-Out: K = number of data points

Performance Metrics:
- Calculate average:
  - Mean squared error
  - R-squared 
  - Mean absolute error

Advantages over single train-test split:
- More stable estimates
- Uses entire dataset for both training and validation
- Reduces impact of random data splitting

**Tarefas para fazer até próxima aula**
- tips
- registros no mobile


