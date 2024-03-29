from pathlib import Path
import argparse

from catboost import CatBoostClassifier

from data_loader import CSVDataLoader
from preprocess import Preprocessor
from pipeline import Pipeline
from optimisation import Optimiser
from utils import Parameter, ParameterType


SUBMISSION_FILE = "submission.csv"


def main(input_path: Path, output_path: Path):

    model = CatBoostClassifier
    parameters = [
        Parameter(name="iterations", type=ParameterType.INTEGER, space=[2500]),
        Parameter(name="depth", type=ParameterType.INTEGER, space=[4, 12]),
        Parameter(name="learning_rate", type=ParameterType.FLOAT, space=[0.01, 0.1]),
        # Parameter(name="random_strength", type=ParameterType.FLOAT, space=[1e-9, 10]),
        # Parameter(name="bagging_temperature", type=ParameterType.FLOAT, space=[0.0, 1.0]),
        # Parameter(name="border_count", type=ParameterType.INTEGER, space=[1, 255]),
        # Parameter(name="l2_leaf_reg", type=ParameterType.FLOAT, space=[2.0, 30.0]),
        Parameter(name="random_state", type=ParameterType.INTEGER, space=[42]),
    ]

    loader = CSVDataLoader(path=input_path)
    preprocessor = Preprocessor({}, {}, {})
    optimiser = Optimiser(model=model, parameters=parameters)

    pipeline = Pipeline(
        model=model, preprocessor=preprocessor, data_loader=loader, output_path=output_path, optimiser=optimiser
    )

    pipeline.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, required=True)
    parser.add_argument('-o', type=str, required=False, default=SUBMISSION_FILE)
    args = parser.parse_args()

    main(Path(args.i), Path(args.o))
