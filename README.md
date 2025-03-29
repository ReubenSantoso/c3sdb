# About

This project centers on enhancing a particle prediction model for ion structure analysis. By integrating classical machine learning models—such as K-Means clustering and Support Vector Regression (SVR) with Graph Neural Networks (soon..), the model aims to predict collision cross-section values with improved accuracy.

The work involved deep analytical preprocessing to improve data quality and processing efficiency. With Python and SQL, custom data cleaning pipelines were built to reduce data redundancy and variance, improving model precision. To further evaluate feature importance and model robustness. 

The primary goals of this project are:

Improve prediction accuracy for particle collision values.

Optimize processing speed and data quality for large, complex datasets.

Offer interpretable model insights.

# Challenges
1. Unstructured and Noisy Ion Collision Datasets
Challenge: The datasets included inconsistent calibration techniques, and missing or extreme outlier values that skewed predictions.

Attempted: Developed data cleaning algorithms using variance thresholds and outlier detection to clean and reduce the dataset.

2. Feature Selection and Dimensionality Reduction
Challenge: High-dimensional datasets created computational bottlenecks and degraded model accuracy.

Attempted: Performed variance analysis with PCA and correlation filtering to eliminate irrelevant features.

3. Interpreting Black-box Models
Challenge: Difficulty in understanding model predictions and variable importance, especially for publication and validation purposes.

Attempting: Integrated a Shapley value-based game theory framework to explain feature contributions to improving interpretability.

4. Ongoing: Graph Structure Construction for GNNs
Challenge: SVR model combinations from our datasets have been exhausted with minimal improvements to reach below 1.6% MSE rate. Now trying graph representations with appropriate edge definitions and node features.

Attempting: Graph Neural Network training on particle behaviors. Also viewing Autoencoders and PCA for dimensionality reduction, and performance of K-Means in deep learning applications.
