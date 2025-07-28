
import os
import pickle
import numpy as np
import pandas as pd

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

    newY = my_model.predict(newX)

    ## Return
    newX["pred"] = newY

    return newX

def predict_and_export(out_path, my_model, pH, aw, temp, model_type = 1):
    pred = grid_prediction(my_model, pH, aw, temp, model_type)

    with open(out_path,'w') as fout:
        pred.to_csv(fout, index = False, lineterminator='\n')


def main():
    ## List the files

    files = os.listdir("out/models")

    ## Load the pickled models

    models = {}

    for each_file in files:

        inpath = os.path.join("out/models/", each_file)

        with open(inpath, 'rb') as infile:
            models[each_file] = pickle.load(infile)

    ## Conditions for the predictions

    pH = np.linspace(4, 9, 50)
    aw = np.linspace(.85, 1, 10)
    temp = np.linspace(7, 18, 50)

    ## Output the grid predictions

    for each_name, each_model in models.items():

        outpath = os.path.join("out/predictions/", each_name+".csv")

        if "first" in each_name:
            model_type = 1
        else:
            model_type = 2

        predict_and_export(outpath, each_model, pH, aw, temp, model_type=model_type)



if __name__ == "__main__": main()