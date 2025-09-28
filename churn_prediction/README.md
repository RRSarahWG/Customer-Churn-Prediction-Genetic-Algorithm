# Customer Churn Prediction using Genetic Algorithm

## Project Overview
This project implements a customer churn prediction system for a telecom company using Genetic Algorithm (GA) for feature selection. The system helps identify customers likely to discontinue services by analyzing various customer attributes and usage patterns.

## Features
- Genetic Algorithm-based feature selection
- Logistic Regression model for churn prediction
- Comprehensive data preprocessing
- Detailed performance visualization
- Model comparison with baseline

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
- Windows:
```bash
.\venv\Scripts\activate
```
- Linux/Mac:
```bash
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure
```
churn_prediction/
├── requirements.txt        # Project dependencies
├── churn_predictor.py     # Main implementation
├── README.md              # Project documentation
└── report/               # Project report and documentation
    └── report.pdf        # Detailed project report
```

## Usage

```python
from churn_predictor import ChurnPredictor

# Initialize predictor
predictor = ChurnPredictor(random_state=42)

# Preprocess data
X, y = predictor.preprocess_data(df)

# Train model with GA feature selection
ga_params = {
    'population_size': 30,
    'generations': 20,
    'mutation_rate': 0.1
}
results, fitness_history = predictor.train(X, y, ga_params)

# Evaluate model
evaluation_results = predictor.evaluate(X_test, y_test, selected_features)
```

## Implementation Details

### 1. Data Preprocessing
- Handling missing values
- Encoding categorical variables
- Feature scaling
- Data splitting for training and testing

### 2. Genetic Algorithm
- Population initialization with binary chromosomes
- Fitness evaluation using logistic regression accuracy
- Tournament selection for parent selection
- Single-point crossover
- Bit-flip mutation
- Elitism preservation

### 3. Model Training
- Baseline model using all features
- GA-optimized model using selected features
- Performance comparison and evaluation

## Results
The GA-optimized model typically achieves:
- Improved accuracy over baseline
- Reduced feature set (15 out of 19 features)
- Better generalization capability
- More interpretable model

## Performance Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a pull request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Dataset source: IBM Telco Customer Churn dataset
- Scikit-learn documentation
- Genetic Algorithm implementation references