import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict
import random
from tqdm import tqdm

class GeneticFeatureSelector:
    """
    A class implementing Genetic Algorithm for feature selection
    """
    def __init__(self, 
                 population_size: int = 50,
                 generations: int = 30,
                 mutation_rate: float = 0.1,
                 elite_size: int = 2,
                 tournament_size: int = 3,
                 random_state: int = 42):
        """
        Initialize the Genetic Algorithm parameters
        """
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.random_state = random_state
        random.seed(random_state)
        np.random.seed(random_state)

    def initialize_population(self, n_features: int) -> List[np.ndarray]:
        """
        Initialize population with random binary vectors
        """
        return [np.random.randint(0, 2, n_features) for _ in range(self.population_size)]

    def fitness_function(self, chromosome: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate fitness using logistic regression accuracy
        """
        if sum(chromosome) == 0:  # If no features selected
            return 0.0
        
        # Select features based on chromosome
        X_selected = X[:, chromosome == 1]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, random_state=self.random_state
        )
        
        # Train and evaluate model
        model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        return accuracy_score(y_test, y_pred)

    def tournament_selection(self, population: List[np.ndarray], 
                           fitnesses: List[float]) -> np.ndarray:
        """
        Select parent using tournament selection
        """
        tournament_idx = random.sample(range(len(population)), self.tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_idx]
        winner_idx = tournament_idx[np.argmax(tournament_fitnesses)]
        return population[winner_idx]

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform crossover between two parents
        """
        point = random.randint(1, len(parent1)-1)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2

    def mutate(self, chromosome: np.ndarray) -> np.ndarray:
        """
        Apply mutation to a chromosome
        """
        for i in range(len(chromosome)):
            if random.random() < self.mutation_rate:
                chromosome[i] = 1 - chromosome[i]  # Flip bit
        return chromosome

    def evolve(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        """
        Main genetic algorithm loop
        """
        n_features = X.shape[1]
        population = self.initialize_population(n_features)
        best_chromosome = None
        best_fitness = 0
        fitness_history = []

        for generation in tqdm(range(self.generations), desc="Genetic Algorithm Progress"):
            # Calculate fitness for each chromosome
            fitnesses = [self.fitness_function(chrom, X, y) for chrom in population]
            
            # Track best solution
            max_fitness_idx = np.argmax(fitnesses)
            if fitnesses[max_fitness_idx] > best_fitness:
                best_fitness = fitnesses[max_fitness_idx]
                best_chromosome = population[max_fitness_idx].copy()
            
            fitness_history.append(best_fitness)

            # Elitism - keep best solutions
            elite_idx = np.argsort(fitnesses)[-self.elite_size:]
            new_population = [population[i].copy() for i in elite_idx]

            # Create new population
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population, fitnesses)
                parent2 = self.tournament_selection(population, fitnesses)
                child1, child2 = self.crossover(parent1, parent2)
                new_population.extend([self.mutate(child1), self.mutate(child2)])

            population = new_population[:self.population_size]

        return best_chromosome, fitness_history


class ChurnPredictor:
    """
    Main class for customer churn prediction
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_selector = None
        self.model = None
        self.feature_names = None

    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the data: handle missing values, encode categorical variables, scale numerical features
        """
        # Remove CustomerID if present
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)

        # Separate features and target
        X = df.drop('Churn', axis=1)
        y = df['Churn'].map({'Yes': 1, 'No': 0})

        # Store feature names
        self.feature_names = X.columns.tolist()

        # Handle categorical variables
        for column in X.select_dtypes(include=['object']):
            self.label_encoders[column] = LabelEncoder()
            X[column] = self.label_encoders[column].fit_transform(X[column])

        # Scale numerical features
        X = self.scaler.fit_transform(X)

        return X, y.values

    def train(self, X: np.ndarray, y: np.ndarray, 
             ga_params: Dict = None) -> Tuple[Dict, List[float]]:
        """
        Train the model using genetic algorithm for feature selection
        """
        # Initialize and run genetic algorithm
        if ga_params is None:
            ga_params = {}
        self.feature_selector = GeneticFeatureSelector(**ga_params)
        best_chromosome, fitness_history = self.feature_selector.evolve(X, y)

        # Train final model with selected features
        X_selected = X[:, best_chromosome == 1]
        self.model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        self.model.fit(X_selected, y)

        # Get selected feature names
        selected_features = [name for i, name in enumerate(self.feature_names) 
                           if best_chromosome[i] == 1]

        return {
            'selected_features': selected_features,
            'n_selected_features': sum(best_chromosome),
            'best_fitness': fitness_history[-1]
        }, fitness_history

    def evaluate(self, X: np.ndarray, y: np.ndarray, 
                selected_features: np.ndarray) -> Dict:
        """
        Evaluate the model performance
        """
        X_selected = X[:, selected_features == 1]
        y_pred = self.model.predict(X_selected)
        
        return {
            'accuracy': accuracy_score(y, y_pred),
            'confusion_matrix': confusion_matrix(y, y_pred),
            'classification_report': classification_report(y, y_pred)
        }


def plot_results(fitness_history: List[float], 
                baseline_accuracy: float,
                final_accuracy: float,
                feature_importances: Dict = None,
                conf_matrix: np.ndarray = None):
    """
    Plot comprehensive results including:
    1. GA evolution over generations
    2. Model performance comparison
    3. Feature importance (if provided)
    4. Confusion matrix (if provided)
    """
    # Create a figure with 2x2 subplots
    fig = plt.figure(figsize=(15, 12))
    
    # 1. GA Evolution Plot
    plt.subplot(2, 2, 1)
    plt.plot(fitness_history, label='GA Feature Selection', color='blue')
    plt.axhline(y=baseline_accuracy, color='r', linestyle='--', 
                label='Baseline Model')
    plt.axhline(y=final_accuracy, color='g', linestyle='--', 
                label='Final Model')
    plt.xlabel('Generation')
    plt.ylabel('Accuracy')
    plt.title('Genetic Algorithm Evolution')
    plt.legend()
    plt.grid(True)

    # 2. Model Performance Comparison
    plt.subplot(2, 2, 2)
    accuracies = [baseline_accuracy, final_accuracy]
    labels = ['Baseline Model\n(All Features)', 'GA Selected\nFeatures']
    bars = plt.bar(labels, accuracies, color=['lightcoral', 'lightgreen'])
    plt.ylabel('Accuracy')
    plt.title('Model Performance Comparison')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')

    # 3. Feature Importance Plot
    if feature_importances:
        plt.subplot(2, 2, 3)
        features = list(feature_importances.keys())
        importance = list(feature_importances.values())
        
        # Sort by importance
        sorted_idx = np.argsort(importance)
        features = [features[i] for i in sorted_idx]
        importance = [importance[i] for i in sorted_idx]
        
        # Plot horizontal bar chart
        plt.barh(range(len(features)), importance)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance Score')
        plt.title('Feature Importance')

    # 4. Confusion Matrix
    if conf_matrix is not None:
        plt.subplot(2, 2, 4)
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

    plt.tight_layout()
    plt.show()

# Test code
if __name__ == "__main__":
    print("Loading and preprocessing the Telco Customer Churn dataset...")
    
    # Load the dataset
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    print(f"Dataset shape: {df.shape}")

    # Initialize the ChurnPredictor
    predictor = ChurnPredictor(random_state=42)
    
    # Preprocess the data
    X, y = predictor.preprocess_data(df)
    print("\nPreprocessing completed.")
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # Train baseline model
    print("\nTraining baseline model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    baseline_model = LogisticRegression(random_state=42, max_iter=1000)
    baseline_model.fit(X_train, y_train)
    baseline_pred = baseline_model.predict(X_test)
    baseline_accuracy = accuracy_score(y_test, baseline_pred)
    print(f"Baseline model accuracy: {baseline_accuracy:.4f}")

    # Train model with GA feature selection
    print("\nTraining model with Genetic Algorithm feature selection...")
    ga_params = {
        'population_size': 30,
        'generations': 20,
        'mutation_rate': 0.1,
        'elite_size': 2,
        'tournament_size': 3,
        'random_state': 42
    }
    
    results, fitness_history = predictor.train(X, y, ga_params)
    
    print("\nGA Feature Selection Results:")
    print(f"Number of selected features: {results['n_selected_features']}")
    print(f"Best fitness score: {results['best_fitness']:.4f}")
    print("\nSelected features:")
    for feature in results['selected_features']:
        print(f"- {feature}")

    # Create chromosome for selected features
    selected_features = np.zeros(X.shape[1])
    for feature in results['selected_features']:
        idx = predictor.feature_names.index(feature)
        selected_features[idx] = 1

    # Evaluate final model
    evaluation_results = predictor.evaluate(X_test, y_test, selected_features)
    print("\nModel Evaluation:")
    print(f"GA-optimized model accuracy: {evaluation_results['accuracy']:.4f}")
    print(f"Improvement over baseline: {(evaluation_results['accuracy'] - baseline_accuracy):.4f}")
    
    print("\nClassification Report:")
    print(evaluation_results['classification_report'])

    # Plot results
    plot_results(fitness_history, baseline_accuracy, evaluation_results['accuracy'])