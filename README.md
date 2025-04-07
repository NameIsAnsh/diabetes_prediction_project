# Diabetes Prediction Project: README

## Project Overview
This project implements a machine learning system for predicting diabetes risk using the Pima Indians Diabetes Dataset. The system includes data preprocessing, exploratory data analysis, model development, evaluation, and a web-based interface for practical application.

## Project Structure
```
diabetes_prediction_project/
├── data/                      # Dataset files
├── models/                    # Trained machine learning models
├── papers/                    # IEEE format research papers
│   ├── conference/            # Conference paper
│   └── review/                # Review paper
├── references/                # Reference papers and resources
├── src/                       # Source code
│   ├── explore_data.py        # Exploratory data analysis
│   ├── preprocess_data.py     # Data preprocessing
│   ├── train_models.py        # Model training
│   ├── evaluate_models.py     # Model evaluation
│   ├── create_interface.py    # Streamlit interface
│   └── run_interface.py       # Interface launcher
├── visualization/             # Data and model visualizations
└── todo.md                    # Project task list
```

## Key Features
1. **Data Preprocessing**: Handling missing values, feature scaling, and data splitting
2. **Exploratory Data Analysis**: Statistical analysis and visualizations of the dataset
3. **Multiple ML Models**: Implementation of Logistic Regression, Random Forest, SVM, KNN, and Neural Networks
4. **Comprehensive Evaluation**: Performance assessment using accuracy, precision, recall, F1-score, and AUC
5. **Feature Importance Analysis**: Identification of key predictors for diabetes risk
6. **Web Interface**: User-friendly Streamlit application for diabetes risk prediction
7. **Research Papers**: IEEE format conference and review papers on diabetes prediction using ML

## Machine Learning Models
The project implements and evaluates six classification algorithms:
- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Neural Network

The K-Nearest Neighbors algorithm achieved the best overall performance with an F1-score of 0.614 and accuracy of 74.46% on the test set.

## Research Papers
Two IEEE format research papers are included:
1. **Conference Paper**: "Early Detection of Diabetes Risk Using Machine Learning: A Comparative Analysis of Classification Algorithms"
2. **Review Paper**: "Machine Learning Approaches for Diabetes Prediction: A Comprehensive Review"

Both papers include comprehensive references to 20 relevant research papers in the field.

## Running the Interface
To run the diabetes prediction interface:
```bash
cd diabetes_prediction_project
python src/run_interface.py
```
This will launch a Streamlit web application where users can input patient data and receive a diabetes risk prediction.

## Requirements
- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- streamlit
- tabulate

## Dataset
The Pima Indians Diabetes Dataset contains medical and demographic data of 768 female patients of Pima Indian heritage, with the following features:
- Pregnancies: Number of times pregnant
- Glucose: Plasma glucose concentration
- BloodPressure: Diastolic blood pressure (mm Hg)
- SkinThickness: Triceps skin fold thickness (mm)
- Insulin: 2-Hour serum insulin (mu U/ml)
- BMI: Body mass index
- DiabetesPedigreeFunction: Diabetes pedigree function
- Age: Age in years
- Outcome: Class variable (0: non-diabetic, 1: diabetic)

## References
A comprehensive list of 20 reference papers is included in the `references/` directory, covering various aspects of diabetes prediction using machine learning.

## Future Work
Potential directions for future development include:
- Exploring advanced ensemble methods and deep learning approaches
- Incorporating additional features such as lifestyle factors and genetic markers
- Addressing class imbalance using techniques like SMOTE
- Validating the models on larger and more diverse datasets
- Conducting prospective studies to evaluate clinical impact
