import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
from itertools import combinations
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_validate
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression

def plot_pairwise_boundaries(X, y, clf, feature_names=None):
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(X.shape[1])]
    
    # Get all pairwise combinations of features
    feature_pairs = list(combinations(range(X.shape[1]), 2))
    
    fig, axes = plt.subplots(1, len(feature_pairs), figsize=(15, 5))
    if len(feature_pairs) == 1:
        axes = [axes]
    
    for idx, (i, j) in enumerate(feature_pairs):
        print ("PRUEBA") 
        print([idx, i, j])
        # Extract 2D data
        X_2d = X.iloc[:, [i, j]]
        
        # Train classifier on 2D data
        clf_2d = type(clf)(**clf.get_params())
        clf_2d.fit(X_2d, y)
        
        # Create decision boundary
        h = 0.02
        x_min, x_max = X_2d.iloc[:, 0].min() - 1, X_2d.iloc[:, 0].max() + 1
        y_min, y_max = X_2d.iloc[:, 1].min() - 1, X_2d.iloc[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))
        
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = clf_2d.predict(mesh_points)
        Z = Z.reshape(xx.shape)
        
        # Plot
        axes[idx].contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
        scatter = axes[idx].scatter(X_2d.iloc[:, 0], X_2d.iloc[:, 1], c=y, cmap='viridis')
        axes[idx].set_xlabel(feature_names[i])
        axes[idx].set_ylabel(feature_names[j])
        axes[idx].set_title(f'{feature_names[i]} vs {feature_names[j]}')
    
    plt.tight_layout()
    plt.show()

df_train = pd.read_csv("training.csv")[["Temperatura", "pH", "Bw", "Crec"]]
df_validation = pd.read_csv("validation.csv")[["Temperatura", "pH", "Bw", "Crec"]]
df = pd.concat([df_train,df_validation])

poly = PolynomialFeatures(degree=2)
X_poly = pd.DataFrame(poly.fit_transform(df.drop("Crec", axis=1)), columns =  poly.get_feature_names_out(["Temperatura", "pH", "Bw"]))
X_poly.rename(columns={"1":"Crec"}, inplace = True)

print(X_poly)
cv = RepeatedKFold(n_splits = 5, n_repeats=2, random_state=123)

X_poly_train, X_poly_test, y_poly_train, y_poly_test = train_test_split(
                X_poly,
                df["Crec"],
                train_size = 0.8,
                random_state = 123,
                shuffle = True
            )

def objective(trial):
        params = {
                "C" : trial.suggest_float('C', 1e-4, 1e4, log=True),
                "penalty" : trial.suggest_categorical('penalty', ['l1', 'l2']),
                "max_iter" : trial.suggest_int('max_iter', 500, 2000),
                "class_weight" : trial.suggest_categorical('class_weight', [None, 'balanced'])
            }


        model = LogisticRegression(
                solver = "saga",
                **params
                )

        model.fit(X_poly_train, y_poly_train)
        predictions = model.predict(X_poly_test)
        score = roc_auc_score(y_poly_test, predictions)
        return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=15, show_progress_bar=False, timeout=60*10)

print('Mejores hiperparámetros:', study.best_params)
print('Mejor score:', study.best_value)

model_best = LogisticRegression(
                solver = "saga",
                **study.best_params)

cv_scores_LogR = cross_validate(model_best, 
                               X = df.drop("Crec", axis=1),
                               y = df["Crec"], 
                               cv = cv,
                               scoring = ("roc_auc", "accuracy"),
                               return_estimator = True,   
                               return_indices = True,
                               return_train_score = True,
                               n_jobs = -1
                               )

plot_pairwise_boundaries(df.drop("Crec", axis=1), df["Crec"], cv_scores_LogR["estimator"][7], feature_names=["Temperatura", "pH", "Bw"])
