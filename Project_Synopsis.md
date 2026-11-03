# Major Project Synopsis

Title: Healthcare Data Analytics and Prediction System

Student Details:
Name: Naman, Nancy, Muskan
Roll No.: 9701, 9702, 9700
Branch Name: Computer Science
Semester: 6th
Mentor: Mr. Nishant Hadda
Project Type: Major Project

## 1. Introduction

Healthcare institutions and insurance organizations generate large amounts of structured data every day, but manual analysis of this data is often slow and inconsistent. Practical data driven systems are needed to identify disease related risk patterns and estimate treatment related costs with better speed and accuracy.

This project builds a machine learning based analytics workflow on two healthcare datasets. The first dataset focuses on heart disease indicators and is prepared for classification use cases. The second dataset focuses on medical insurance charges and is prepared for regression use cases. The project is implemented using Python notebooks with strong emphasis on data cleaning, feature preparation, and statistical analysis.

The overall goal is to create a simple, understandable, and extensible analytics system that can be used as a strong academic major project and later upgraded into a complete predictive application.

## 2. Problem Statement

In many environments, healthcare and insurance data is available but not used effectively due to poor preprocessing and lack of structured analysis.

Key issues include:
- Inconsistent and noisy data values (such as duplicates, zero placeholders, and mixed categorical formats).
- Difficulty in identifying the most influential features related to heart risk and insurance cost.
- Lack of a clean pipeline that converts raw records into model ready datasets.
- Limited interpretability when analysis is done without proper visual and statistical methods.

There is a need for a practical and reproducible data pipeline that improves data quality, supports meaningful feature extraction, and prepares robust inputs for machine learning models.

## 3. Objectives

The main objectives of this project are:
- To perform exploratory data analysis on heart and insurance datasets.
- To clean and preprocess numerical and categorical features.
- To encode categorical attributes into machine readable format.
- To scale selected numerical features using standardization.
- To engineer additional features such as BMI category for better interpretation.
- To evaluate feature relevance using Pearson correlation and chi square testing.
- To produce final model ready datasets for classification and regression tasks.

## 4. Key Features

1. Dual Dataset Workflow:
- Heart disease analysis workflow using [Heart.ipynb](Heart.ipynb) and [heart.csv](heart.csv).
- Insurance cost analysis workflow using [insurance.ipynb](insurance.ipynb) and [insurance.csv](insurance.csv).

2. Data Cleaning:
- Duplicate detection and removal.
- Handling of invalid zero values in clinical columns.
- Data type normalization.

3. Feature Transformation:
- One hot encoding of categorical features.
- Binary mapping for selected variables.
- Column renaming for improved readability.

4. Feature Engineering:
- Creation of bmi_category groups.
- Selection of final features for downstream modeling.

5. Statistical Analysis:
- Correlation analysis for numeric feature importance.
- Chi square testing for categorical feature significance.

6. Visualization Support:
- Histograms, count plots, box plots, violin plots, and heatmaps for interpretability.

## 5. Methodology and Tools

Development Approach:
- The project is executed in stages: data understanding, cleaning, transformation, feature extraction, and statistical screening.

Process Flow:
1. Requirement understanding and dataset inspection.
2. Exploratory data analysis using descriptive statistics and visual plots.
3. Data cleaning and handling of inconsistent records.
4. Categorical encoding and numerical scaling.
5. Feature engineering and relevance testing.
6. Preparation of processed data for model development.
7. Documentation of workflow and outcomes.

Technology Stack:
- Language: Python
- Environment: Jupyter Notebook
- Libraries: NumPy, Pandas, Matplotlib, Seaborn, scikit learn, SciPy
- Helper package used in analysis: sheryanalysis

## 6. Software and Hardware Requirements

Software Requirements:
- Operating System: Windows 10 or above
- Python: 3.x
- Notebook Runtime: Jupyter environment (Anaconda or virtual environment)
- Required Python packages:
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- scipy

Hardware Requirements:
- Processor: Intel i3 or equivalent and above
- RAM: Minimum 4 GB (8 GB recommended)
- Storage: At least 2 GB free for environment, notebooks, and outputs

## 7. Expected Outcomes

- Well structured and cleaned datasets for both healthcare domains.
- Improved understanding of major factors influencing heart disease and insurance charges.
- Reusable preprocessing pipeline for academic demonstration.
- A strong foundation for adding predictive models and evaluation in the next phase.
- Better interpretability through visual and statistical analysis.

## 8. Future Scope

Future enhancements include:
- Train and compare classification models for heart disease prediction.
- Train and compare regression models for insurance charge prediction.
- Add cross validation, hyperparameter tuning, and error analysis.
- Integrate explainable AI methods (for example SHAP based insights).
- Build a simple web interface for user level prediction.
- Package workflow into modular scripts for easier deployment.

## 9. References

- Official documentation of Pandas, NumPy, Matplotlib, and Seaborn.
- scikit-learn documentation for preprocessing and modeling workflows.
- SciPy documentation for Pearson and chi square statistical tests.
- Publicly available heart disease and insurance dataset references used in notebooks.

## 10. Conclusion

This project provides a practical and well organized machine learning data pipeline for two relevant healthcare analytics problems: heart disease risk analysis and insurance charge analysis. By focusing on preprocessing quality, feature engineering, and statistical validation, the system ensures that future model training can be more reliable and interpretable. The work is suitable as a major academic project and can be extended into a complete prediction platform in subsequent phases.
