
from skopt import BayesSearchCV
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier, Perceptron
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier, XGBRFClassifier

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pickle



## Data import ---------------------------------------------------------------------------------------------------------

def import_data():

    df_train = pd.read_csv("training.csv")
    df_validation = pd.read_csv("validation.csv")
    df_all = pd.concat([df_train,df_validation])

    Y = df_all['Crec'].values
    df_all = df_all.drop(["Temperatura.1", "pH.1", "Aw.1", "Crec", "Pest"], axis = 1)

    return df_all, Y

def split_transform(X, y, test_size = 0.3):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    ## Remove the interactions

    X_train_1 = X_train.iloc[:, [0, 1, 2]]
    X_test_1 = X_test.iloc[:, [0, 1, 2]]

    return X_train, X_test, y_train, y_test, X_train_1, X_test_1

## Error indices -------------------------------------------------------------------------------------------------------

def export_errors_indices(model, X_train, y_train, X_test, y_test, cv, out_path):
    train_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")

    y_pred = model.predict(X_test)
    test_score = roc_auc_score(y_test, y_pred)

    out = pd.DataFrame()
    out["scores"] = np.append(train_scores, test_score)
    aa = ["train"] * cv
    aa.append("test")
    out["type"] = aa

    with open(out_path, 'w') as outfile:
        out.to_csv(outfile, index=False, lineterminator='\n')

    return out

## Prediction and export -----------------------------------------------------------------------------------------------

def grid_prediction(my_model, pH, aw, temp, model_type = 1):

    bw = np.sqrt(1 - aw)
    # Create a grid of all combinations
    A, B, C = np.meshgrid(temp, pH, bw)
    newX = np.column_stack([A.ravel(), B.ravel(), C.ravel()])

    newX = pd.DataFrame(newX, columns = ["Temperatura", "pH", "Bw"])

    ## Add higher order terms if needed
    if model_type == 2:
        newX["T2"] = newX["Temperatura"]**2
        newX["pH2"] = newX["pH"] ** 2
        newX["Bw2"] = newX["Bw"] ** 2
        newX["TxpH"] = newX["Temperatura"] * newX["pH"]
        newX["TxBw"] = newX["Temperatura"] * newX["Bw"]
        newX["pHxBw"] = newX["pH"] * newX["Bw"]

    ## Make prediction

    newY = my_model.best_estimator_.predict(newX)

    ## Return
    newX["pred"] = newY

    return newX

def predict_and_export(out_path, my_model, pH, aw, temp, model_type = 1):
    pred = grid_prediction(my_model, pH, aw, temp, model_type)

    with open(out_path,'w') as fout:
        pred.to_csv(fout, index = False, lineterminator='\n')

## Models --------------------------------------------------------------------------------------------------------------

def fit_SVM_linear(X_train, y_train, X_test, y_test,
                export_model=False, export_indices=False,
                niter=20,
                cv=5):

    ## Pipeline

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC())
    ])

    ## Optimization problem

    opt = BayesSearchCV(
        pipeline,
        {
            'svm__C': (1e-3, 10, 'uniform'),
            'svm__kernel': ['linear'],
            'svm__gamma': (1e-9, 1e3, 'uniform'),

        },
        n_iter=niter,
        cv=cv
    )

    model = opt.fit(X_train, y_train)
    model_best = model.best_estimator_

    ## Export the model if requested

    if export_model:
        with open(export_model, 'wb') as outfile:
            pickle.dump(model_best, outfile)

    ## Calculate and export the indices if requested

    if export_indices:
        export_errors_indices(model, X_train, y_train, X_test, y_test, cv, export_indices)

    ## Return

    return model_best

def fit_RF(X_train, y_train, X_test, y_test,
                export_model=False, export_indices=False,
                niter=20,
                cv=5):

    ## Pipeline

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('RFC', RandomForestClassifier())
    ])

    ## Optimization problem

    opt = BayesSearchCV(
        pipeline,
        {
            'RFC__max_depth': [ int(x) for x in range(5,20) ],
            'RFC__max_features': [ int(x) for x in range(5,20) ],
            'RFC__n_estimators': [ int(x) for x in range(20,50) ],

        },
        n_iter=niter,
        cv=cv
    )

    model = opt.fit(X_train, y_train)
    model_best = model.best_estimator_

    ## Export the model if requested

    if export_model:
        with open(export_model, 'wb') as outfile:
            pickle.dump(model_best, outfile)

    ## Calculate and export the indices if requested

    if export_indices:
        export_errors_indices(model, X_train, y_train, X_test, y_test, cv, export_indices)

    ## Return

    return model_best
def fit_lightgbmc(X_train, y_train, X_test, y_test,
                export_model=False, export_indices=False,
                niter=20,
                cv=5):

    ## Pipeline

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('LGBMC', LGBMClassifier())
    ])

    ## Optimization problem

    opt = BayesSearchCV(
        pipeline,
        {
            'LGBMC__n_estimators': [ int(x) for x in range(1,10) ],
            'LGBMC__max_depth': [ int(x) for x in range(5,10) ],
            'LGBMC__learning_rate': (1e-4, 1e-1, 'uniform'),
            'LGBMC__subsample': (.01, 1, 'uniform'),
            'LGBMC__boosting_type': ['gbdt', 'dart'],
            'LGBMC__num_leaves': [ int(x) for x in range(1000,2000) ]

        },
        n_iter=niter,
        cv=cv
    )

    model = opt.fit(X_train, y_train)
    model_best = model.best_estimator_

    ## Export the model if requested

    if export_model:
        with open(export_model, 'wb') as outfile:
            pickle.dump(model_best, outfile)

    ## Calculate and export the indices if requested

    if export_indices:
        export_errors_indices(model, X_train, y_train, X_test, y_test, cv, export_indices)

    ## Return

    return model_best
def fit_multinomialNB(X_train, y_train, X_test, y_test,
                export_model=False, export_indices=False,
                niter=20,
                cv=5):

    ## Pipeline

    pipeline = Pipeline([
        # ('scaler', StandardScaler()),
        ('MNB', MultinomialNB())
    ])

    ## Optimization problem

    opt = BayesSearchCV(
        pipeline,
        {
            "MNB__alpha": (.1, 10, 'uniform'),
            "MNB__fit_prior": [True, False]

        },
        n_iter=niter,
        cv=cv
    )

    model = opt.fit(X_train, y_train)
    model_best = model.best_estimator_

    ## Export the model if requested

    if export_model:
        with open(export_model, 'wb') as outfile:
            pickle.dump(model_best, outfile)

    ## Calculate and export the indices if requested

    if export_indices:
        export_errors_indices(model, X_train, y_train, X_test, y_test, cv, export_indices)

    ## Return

    return model_best
def fit_perceptron(X_train, y_train, X_test, y_test,
                export_model=False, export_indices=False,
                niter=20,
                cv=5):

    ## Pipeline

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('PR', Perceptron())
    ])

    ## Optimization problem

    opt = BayesSearchCV(
        pipeline,
        {
            "PR__penalty": [None, "l2", "l1"],
            "PR__eta0": (1e-3, 1e-1, 'log-uniform'),
            # "PR__max_iter": [1000, 2000],
            "PR__class_weight": [None, "balanced"]

        },
        n_iter=niter,
        cv=cv
    )

    model = opt.fit(X_train, y_train)
    model_best = model.best_estimator_

    ## Export the model if requested

    if export_model:
        with open(export_model, 'wb') as outfile:
            pickle.dump(model_best, outfile)

    ## Calculate and export the indices if requested

    if export_indices:
        export_errors_indices(model, X_train, y_train, X_test, y_test, cv, export_indices)

    ## Return

    return model_best
def fit_SGD(X_train, y_train, X_test, y_test,
                export_model=False, export_indices=False,
                niter=20,
                cv=5):

    ## Pipeline

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('SGD', SGDClassifier())
    ])

    ## Optimization problem

    opt = BayesSearchCV(
        pipeline,
        {
            "SGD__loss": ['hinge', 'log_loss', 'modified_huber'],
            "SGD__penalty": ['l2', 'l1', 'elasticnet'],
            "SGD__alpha": (1e-6, 1e-1, 'log-uniform')
        },
        n_iter=niter,
        cv=cv
    )

    model = opt.fit(X_train, y_train)
    model_best = model.best_estimator_

    ## Export the model if requested

    if export_model:
        with open(export_model, 'wb') as outfile:
            pickle.dump(model_best, outfile)

    ## Calculate and export the indices if requested

    if export_indices:
        export_errors_indices(model, X_train, y_train, X_test, y_test, cv, export_indices)

    ## Return

    return model_best
def fit_MLP(X_train, y_train, X_test, y_test,
                export_model=False, export_indices=False,
                niter=20,
                cv=5):

    ## Pipeline

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('MLP', MLPClassifier())
    ])

    ## Optimization problem

    opt = BayesSearchCV(
        pipeline,
        {
            "MLP__hidden_layer_sizes": [(50,), (100,), (100, 50), (200, 100)],
            "MLP__alpha": (1e-6, 1e-1, 'uniform'),
            "MLP__learning_rate_init": (1e-5, 1e-1, 'uniform'),
            "MLP__activation": ['relu', 'tanh'],
            "MLP__solver": ['adam']
        },
        n_iter=niter,
        cv=cv
    )

    model = opt.fit(X_train, y_train)
    model_best = model.best_estimator_

    ## Export the model if requested

    if export_model:
        with open(export_model, 'wb') as outfile:
            pickle.dump(model_best, outfile)

    ## Calculate and export the indices if requested

    if export_indices:
        export_errors_indices(model, X_train, y_train, X_test, y_test, cv, export_indices)

    ## Return

    return model_best
def fit_XGB(X_train, y_train, X_test, y_test,
                export_model=False, export_indices=False,
                niter=20,
                cv=5):

    ## Pipeline

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('XGB', XGBClassifier())
    ])

    ## Optimization problem

    opt = BayesSearchCV(
        pipeline,
        {
            'XGB__n_estimators': list(range(2, 6)),
            'XGB__max_depth': list(range(3, 12)),
            # 'XGB__scale_pos_weight': [1, 2, 5],
            'XGB__learning_rate': (1e-4, 1e-1, 'log-uniform')
            # 'XGB__reg_lambda': [1e-4, 1e-2, 1e-1],
            # 'XGB__reg_alpha': [1e-4, 1e-1],
            # 'XGB__subsample': [.1, 1]
        },
        n_iter=niter,
        cv=cv
    )

    model = opt.fit(X_train, y_train)
    model_best = model.best_estimator_

    ## Export the model if requested

    if export_model:
        with open(export_model, 'wb') as outfile:
            pickle.dump(model_best, outfile)

    ## Calculate and export the indices if requested

    if export_indices:
        export_errors_indices(model, X_train, y_train, X_test, y_test, cv, export_indices)

    ## Return

    return model_best
def fit_XGBRFC(X_train, y_train, X_test, y_test,
                export_model=False, export_indices=False,
                niter=20,
                cv=5):

    ## Pipeline

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('XGBRFC', XGBClassifier())
    ])

    ## Optimization problem

    opt = BayesSearchCV(
        pipeline,
        {
            'XGBRFC__n_estimators': (10, 1000, 'log-uniform'),
            'XGBRFC__max_depth': (10, 50, 'log-uniform'),
            'XGBRFC__reg_lambda': (1e-5, 1, 'log-uniform'),
            'XGBRFC__reg_alpha': (1e-5, 1, 'log-uniform'),
            'XGBRFC__gamma': (1e-5, 1, 'log-uniform'),
            'XGBRFC__subsample': (.1, 1, 'log-uniform'),
            'XGBRFC__colsample_bynode': (.1, 1, 'log-uniform'),
            'XGBRFC__learning_rate': (1e-4, 1e-2, 'log-uniform'),
        },
        n_iter=niter,
        cv=cv
    )

    model = opt.fit(X_train, y_train)
    model_best = model.best_estimator_

    ## Export the model if requested

    if export_model:
        with open(export_model, 'wb') as outfile:
            pickle.dump(model_best, outfile)

    ## Calculate and export the indices if requested

    if export_indices:
        export_errors_indices(model, X_train, y_train, X_test, y_test, cv, export_indices)

    ## Return

    return model_best
def fit_SVM_rbf(X_train, y_train, X_test, y_test,
                export_model=False, export_indices=False,
                niter=20,
                cv=5):

    ## Pipeline

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC())
    ])

    ## Optimization problem

    opt = BayesSearchCV(
        pipeline,
        {
            'svm__C': (1e-6, 1e+6, 'log-uniform'),
            'svm__gamma': (1e-6, 1e+1, 'log-uniform'),
            # 'degree': (1, 8),  # integer valued parameter
            # 'kernel': ['linear', 'poly', 'rbf'],  # categorical parameter
            'svm__kernel': ['rbf'],  # categorical parameter
        },
        n_iter=niter,
        cv=cv
    )

    model = opt.fit(X_train, y_train)
    model_best = model.best_estimator_

    ## Export the model if requested

    if export_model:
        with open(export_model, 'wb') as outfile:
            pickle.dump(model_best, outfile)

    ## Calculate and export the indices if requested

    if export_indices:
        export_errors_indices(model, X_train, y_train, X_test, y_test, cv, export_indices)

    ## Return

    return model_best

def fit_toxic(X_train, y_train, X_test, y_test,
                export_model=False, export_indices=False,
                niter=20,
                cv=5):

    ## Pipeline

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('toxic', PassiveAggressiveClassifier())
    ])

    ## Optimization problem

    opt = BayesSearchCV(
        pipeline,
        {
            "toxic__C": (1e-4, 1e+2, 'log-uniform'),
            "toxic__loss": ['hinge', 'squared_hinge'],
            "toxic__fit_intercept": [True, False],
            "toxic__shuffle": [True],
            "toxic__class_weight": [None, "balanced"],
            "toxic__average": [False, True]
        },
        n_iter=niter,
        cv=cv
    )

    model = opt.fit(X_train, y_train)
    model_best = model.best_estimator_

    ## Export the model if requested

    if export_model:
        with open(export_model, 'wb') as outfile:
            pickle.dump(model_best, outfile)

    ## Calculate and export the indices if requested

    if export_indices:
        export_errors_indices(model, X_train, y_train, X_test, y_test, cv, export_indices)

    ## Return

    return model_best

## Main functions ------------------------------------------------------------------------------------------------------

def main(first_order=True):

    ## Import and split the data

    X, y = import_data()

    X_train, X_test, y_train, y_test, X_train_1, X_test_1 = split_transform(X, y)

    ## Fit first order models

    if first_order == True:
        print("FITTING FIRST ORDER")
        
        print("SVM linear")

        fit_SVM_linear(X_train_1, y_train, X_test_1, y_test,
                    export_model = "../models_datos_Alberto/SVMlin_first.p",
                    export_indices = "out/errors/SVMlin_first.csv",
                    niter=20,
                    cv=5
                    )

        print("RF")

        fit_RF(X_train_1, y_train, X_test_1, y_test,
                    export_model = "../models_datos_Alberto/RF_first.p",
                    export_indices = "out/errors/RF_first.csv",
                    niter=20,
                    cv=5
                    )

        print("LGBM")

        fit_lightgbmc(X_train_1, y_train, X_test_1, y_test,
                    export_model = "../models_datos_Alberto/LGBM_first.p",
                    export_indices = "out/errors/LGBM_first.csv",
                    niter=20,
                    cv=5
                    )

        print("Multinomial NB")

        fit_multinomialNB(X_train_1, y_train, X_test_1, y_test,
                    export_model = "../models_datos_Alberto/MNB_first.p",
                    export_indices = "out/errors/MNB_first.csv",
                    niter=20,
                    cv=5
                    )

        print("Perceptron")

        fit_perceptron(X_train_1, y_train, X_test_1, y_test,
                    export_model = "../models_datos_Alberto/perceptron_first.p",
                    export_indices = "out/errors/perceptron_first.csv",
                    niter=20,
                    cv=5
                    )

        print("SGD")

        fit_SGD(X_train_1, y_train, X_test_1, y_test,
                    export_model = "../models_datos_Alberto/SGD_first.p",
                    export_indices = "out/errors/SGD_first.csv",
                    niter=20,
                    cv=5
                    )

        print("MLP")
        '''
        fit_MLP(X_train_1, y_train, X_test_1, y_test,
                    export_model = "../models_datos_Alberto/MLP_first.p",
                    export_indices = "out/errors/MLP_first.csv",
                    niter=20,
                    cv=5
                    )
        '''
        print("XGB")

        fit_XGB(X_train_1, y_train, X_test_1, y_test,
                    export_model = "../models_datos_Alberto/XGB_first.p",
                    export_indices = "out/errors/XGB_first.csv",
                    niter=20,
                    cv=5
                    )

        print("XGBRFC")

        fit_XGBRFC(X_train_1, y_train, X_test_1, y_test,
                    export_model = "../models_datos_Alberto/XGBRFC_first.p",
                    export_indices = "out/errors/XGBRFC_first.csv",
                    niter=20,
                    cv=5
                    )

        print("Passive Agressive")
        fit_toxic(X_train_1, y_train, X_test_1, y_test,
                    export_model = "../models_datos_Alberto/PassAg_first.p",
                    export_indices = "out/errors/PassAg_first.csv",
                    niter=20,
                    cv=5
                    )

        print("SVM with radial")
        fit_SVM_rbf(X_train_1, y_train, X_test_1, y_test,
                    export_model = "../models_datos_Alberto/SVMrbf_first.p",
                    export_indices = "out/errors/SVMrbf_first.csv",
                    niter=20,
                    cv=5
                    )

    ## Fit second order models

    print("FITTING SECOND ORDER")

    print("SVM linear")

    fit_SVM_linear(X_train, y_train, X_test, y_test,
                   export_model="../models_datos_Alberto/SVMlin_second.p",
                   export_indices="out/errors/SVMlin_second.csv",
                   niter=20,
                   cv=5
                   )

    print("RF")

    fit_RF(X_train, y_train, X_test, y_test,
           export_model="../models_datos_Alberto/RF_second.p",
           export_indices="out/errors/RF_second.csv",
           niter=2,
           cv=5
           )

    print("LGBM")

    fit_lightgbmc(X_train, y_train, X_test, y_test,
                  export_model="../models_datos_Alberto/LGBM_second.p",
                  export_indices="out/errors/LGBM_second.csv",
                  niter=20,
                  cv=5
                  )

    print("Multinomial NB")

    fit_multinomialNB(X_train, y_train, X_test, y_test,
                      export_model="../models_datos_Alberto/MNB_second.p",
                      export_indices="out/errors/MNB_second.csv",
                      niter=20,
                      cv=5
                      )

    print("Perceptron")

    fit_perceptron(X_train, y_train, X_test, y_test,
                   export_model="../models_datos_Alberto/perceptron_second.p",
                   export_indices="out/errors/perceptron_second.csv",
                   niter=20,
                   cv=5
                   )

    print("SGD")

    fit_SGD(X_train, y_train, X_test, y_test,
            export_model="../models_datos_Alberto/SGD_second.p",
            export_indices="out/errors/SGD_second.csv",
            niter=20,
            cv=5
            )
    '''
    
    print("MLP")

    fit_MLP(X_train, y_train, X_test, y_test,
            export_model="../models_datos_Alberto/MLP_second.p",
            export_indices="out/errors/MLP_second.csv",
            niter=2,
            cv=5
            )
    '''
    
    print("XGB")

    fit_XGB(X_train, y_train, X_test, y_test,
            export_model="../models_datos_Alberto/XGB_second.p",
            export_indices="out/errors/XGB_second.csv",
            niter=20,
            cv=5
            )

    print("XGBRFC")

    fit_XGBRFC(X_train, y_train, X_test, y_test,
               export_model="../models_datos_Alberto/XGBRFC_second.p",
               export_indices="out/errors/XGBRFC_second.csv",
               niter=20,
               cv=5
               )

    print("Passive Agressive")
    fit_toxic(X_train, y_train, X_test, y_test,
              export_model="../models_datos_Alberto/PassAg_second.p",
              export_indices="out/errors/PassAg_second.csv",
              niter=20,
              cv=5
              )

    print("SVM with radial")
    fit_SVM_rbf(X_train, y_train, X_test, y_test,
                export_model="../models_datos_Alberto/SVMrbf_second.p",
                export_indices="out/errors/SVMrbf_second.csv",
                niter=20,
                cv=5
                )

if __name__ == "__main__": main()

