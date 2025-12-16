import pandas as pd
import numpy as np
import os
import sys
import pickle
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from bayesian_hyper import import_data, export_errors_indices
from prediction import grid_prediction, predict_and_export, main

pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

#new_data = pd.read_excel(r"C:\Users\dres2\OneDrive - Universidad Politécnica de Cartagena\Documentos\repositories\datos_Alberto\new_data\Experimento_modificaciones_16ºC.xlsx")
#path_data = r"C:\Users\dres2\OneDrive - Universidad Politécnica de Cartagena\Documentos\repositories\datos_Alberto\new_data"

new_data = pd.read_excel("new_data/Experimento_modificaciones_16ºC.xlsx")
path_data= "new_data"

df_list = {}

# reading new data

for dataset_ in os.listdir(path_data):
    
    if dataset_ != "Experimento_modificaciones_16ºC.xlsx":
        # Temperatura	pH	Aw	p	Crec
        path = os.path.join(path_data, dataset_)
        dataset = pd.read_csv(path, sep= "\t", encoding='latin-1')
    
    else:
        dataset = new_data
        dataset.rename(columns={"Temp" : "Temperatura"}, inplace=True)


    dataset.loc[: , "Bw"] = np.sqrt(1 - dataset.loc[: , "Aw"])
    dataset.loc[: , "T2"] = dataset.loc[: , "Temperatura"]**2
    dataset.loc[: , "pH2"] = dataset.loc[: , "pH"] ** 2
    dataset.loc[: , "Aw2"] = dataset.loc[: , "Aw"] ** 2
    dataset.loc[: , "Bw2"] = dataset.loc[: , "Bw"] ** 2
    dataset.loc[: , "TxpH"] = dataset.loc[: , "Temperatura"] * dataset.loc[: , "pH"]
    dataset.loc[: , "TxAw"] = dataset.loc[: , "Temperatura"] * dataset.loc[: , "Aw"]
    dataset.loc[: , "pHxAw"] = dataset.loc[: , "pH"] * dataset.loc[: , "Aw"]
    dataset.loc[: , "TxBw"] = dataset.loc[: , "Temperatura"] * dataset.loc[: , "Bw"]
    dataset.loc[: , "pHxBw"] = dataset.loc[: , "pH"] * dataset.loc[: , "Bw"]

#Index(['Temperatura', 'pH', 'Aw', 'Bw', 'T2', 'pH2', 'Aw2', 'Bw2', 'TxpH',
#       'TxAw', 'pHxAw', 'TxBw', 'pHxBw'],

    df_list[dataset_] = dataset


#path_models = r"C:\Users\dres2\OneDrive - Universidad Politécnica de Cartagena\Documentos\repositories\models_datos_Alberto\models\Second execution"
path_models = "models"
models_ = ["SGD_second.p", "perceptron_second.p", "PassAg_second.p"] # os.listdir(path_models)

list_models = {}

# loading models to be incremented

for model in models_:
    path = os.path.join(path_models, model)
    
    with open(path, "rb") as f:
        model_loaded = pickle.load(f) 
    list_models[model] = model_loaded

df_train = df_list["Experimento_modificaciones_16ºC.xlsx"]

Y_new = df_train["Crec"].values
X_new = df_train.drop(["Crec", "Muestra"], axis = 1)

# array(['Temperatura', 'pH', 'Aw', 'Bw', 'T2', 'pH2', 'Aw2', 'Bw2', 'TxpH', 'TxAw', 'pHxAw', 'TxBw', 'pHxBw'], dtype=object)
# X_1 = X.iloc[:,[0,1,2]]

new_SGD = list_models["SGD_second.p"].named_steps["SGD"].partial_fit(X_new, 1-Y_new)
print(new_SGD)
new_perceptron = list_models["perceptron_second.p"].named_steps["PR"].partial_fit(X_new, 1-Y_new)
print(new_perceptron)
new_PassAg = list_models["PassAg_second.p"].named_steps["toxic"].partial_fit(X_new, 1-Y_new)
print(new_PassAg)


def split_transform(X, y, test_size = 0.3):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test


def partial_fit_pipeline(pipelineObj, modelname, X_new, Y_new):
    
    scaler = pipelineObj.named_steps["scaler"]
    X_new_scaled = scaler.transform(X_new)
    X, y = import_data()
    
    X_train, X_test, y_train, y_test = split_transform(scaler.transform(X),y)
    model = pipelineObj.named_steps[modelname].partial_fit(X_new_scaled, Y_new)
    
    X_train = pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_new_scaled)], axis=0)
    y_train = pd.Series(np.append(y_train, Y_new), dtype="category")    
    
    cv = 5
    savepath = "out/errors/partial_fit_1/"+modelname+"_partial_fit.csv"
    export_errors_indices(model, X_train, y_train, X_test, y_test, cv, savepath)
    
    X_train.columns = X.columns
    
    pH = np.linspace(4, 9, 50)
    aw = np.linspace(.85, 1, 10)
    temp = np.linspace(7, 18, 50)

    out_path = "out/predictions/partial_fit"
    predict_and_export(out_path= out_path + modelname + "_predictions.csv", 
                        my_model = model, 
                        pH = pH,
                        bw = aw,
                        temp = temp,
                        scaler = scaler,
                        model_type=2)
    
    export_model = out_path + modelname + "_partial_fit.p"
    with open(export_model, 'wb') as outfile:
        pickle.dump(model, outfile)
    
    return model

SGD_second_partial = partial_fit_pipeline(list_models["SGD_second.p"], "SGD", X_new, Y_new)
perceptron_second_partial = partial_fit_pipeline(list_models["perceptron_second.p"], "PR", X_new, Y_new)
PassAg_second_partial = partial_fit_pipeline(list_models["PassAg_second.p"], "toxic", X_new, Y_new)

