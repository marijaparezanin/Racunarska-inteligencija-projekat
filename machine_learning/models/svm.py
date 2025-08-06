from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, recall_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV


def train_and_evaluate_best_svm(X, y, kernels=['linear', 'rbf'], C_range=[0.1, 1, 10]):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


    param_grid = {
        'kernel': kernels,
        'C': C_range
    }

    random_search = RandomizedSearchCV(
        SVC(probability=True),
        param_distributions=param_grid,
        n_iter=2,  # Try just 4 random combos
        cv=2,
        scoring='recall',
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)

    print("\nBest Parameters:", random_search.best_params_)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # Optional: plot heatmap of results
    results = random_search.cv_results_
    scores = results['mean_test_score']
    kernels = param_grid['kernel']
    Cs = param_grid['C']
    
    score_matrix = np.array(scores).reshape(len(Cs), len(kernels))
    plt.figure(figsize=(8, 6))
    for i, kernel in enumerate(kernels):
        plt.plot(Cs, score_matrix[:, i], label=f'Kernel: {kernel}')
    plt.xscale('log')
    plt.xlabel('C (log scale)')
    plt.ylabel('Mean CV Recall')
    plt.title('SVM Hyperparameter Tuning')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return best_model, y_test, y_pred
