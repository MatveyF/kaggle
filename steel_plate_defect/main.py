from pathlib import Path
import argparse

from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler

from data_loader import CSVDataLoader
from preprocess import Preprocessor
from pipeline import Pipeline
from optimisation import Optimiser
from utils import Parameter, ParameterType


SUBMISSION_FILE = "submission.csv"


def main(input_path: Path, output_path: Path):

    model = CatBoostClassifier

    # Parameter ranges for optimisation
    # parameters = [
    #     Parameter(name="iterations", type=ParameterType.INTEGER, space=[1000, 2000]),
    #     Parameter(name="n_estimators", type=ParameterType.INTEGER, space=[200, 1000]),
    #     Parameter(name="depth", type=ParameterType.INTEGER, space=[3, 10]),
    #     Parameter(name="learning_rate", type=ParameterType.FLOAT, space=[0.01, 0.1]),
    #     Parameter(name="l2_leaf_reg", type=ParameterType.FLOAT, space=[0.0, 10.0]),
    #     Parameter(name="subsample", type=ParameterType.FLOAT, space=[0.1, 1.0]),
    #     Parameter(name="random_state", type=ParameterType.INTEGER, space=[42]),
    #     Parameter(name="verbose", type=ParameterType.CATEGORICAL, space=[False]),
    # ]

    # these were the best found parameters (note, no ranges, single values only)
    parameters = [
        Parameter(name="iterations", type=ParameterType.INTEGER, space=[1392]),
        Parameter(name="depth", type=ParameterType.INTEGER, space=[4]),
        Parameter(name="learning_rate", type=ParameterType.FLOAT, space=[0.06498]),
        Parameter(name="l2_leaf_reg", type=ParameterType.FLOAT, space=[6.24262]),
        Parameter(name="random_state", type=ParameterType.INTEGER, space=[42]),
        Parameter(name="verbose", type=ParameterType.CATEGORICAL, space=[False]),
    ]

    scalers = {
        "Pixels_Areas": StandardScaler(),
        "Steel_Plate_Thickness": StandardScaler(),
        "Empty_Index": StandardScaler(),
    }

    loader = CSVDataLoader(path=input_path)
    preprocessor = Preprocessor(scalers=scalers, imputers={}, encoders={}, feature_engineering=True)
    optimiser = Optimiser(model=model, parameters=parameters, n_trials=1)

    pipeline = Pipeline(
        model=model, preprocessor=preprocessor, data_loader=loader, output_path=output_path, optimiser=optimiser
    )

    pipeline.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str, required=True)
    parser.add_argument("-o", type=str, required=False, default=SUBMISSION_FILE)
    args = parser.parse_args()

    main(Path(args.i), Path(args.o))
