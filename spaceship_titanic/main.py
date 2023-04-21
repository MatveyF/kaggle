# Imports
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GridSearchCV

from preprocess import preprocess_data


def main(input_path: Path):
    orig_train = pd.read_csv(input_path / "train.csv")
    orig_test = pd.read_csv(input_path / "test.csv")

    train = preprocess_data(orig_train)
    test = preprocess_data(orig_test)

    from catboost import CatBoostClassifier

    y_train = train["Transported"]
    X_train = train.drop(columns=["Transported"])

    param_grid = {
        'learning_rate': [0.01, 0.1, 0.2],
        'iterations': [500, 1000],
        'depth': [4, 6, 8],
        'l2_leaf_reg': [1, 3, 5],
    }

    model = CatBoostClassifier()

    # Perform Grid Search
    grid_search = GridSearchCV(model, param_grid, cv=4, n_jobs=-1, verbose=3)
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_

    # Train your final model with the best parameters
    final_model = CatBoostClassifier(**best_params)
    final_model.fit(X_train, y_train)

    y_pred = final_model.predict(test)

    result = pd.DataFrame({"PassengerId": orig_test["PassengerId"], "Transported": y_pred})

    # result["Transported"] = result["Transported"].apply(lambda x: x in [1, True])

    result.to_csv("submission.csv", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, required=True)
    # parser.add_argument('-o', type=str, required=True)
    args = parser.parse_args()

    main(Path(args.i))
