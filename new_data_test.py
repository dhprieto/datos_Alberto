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
