from pathlib import Path
import argparse

from catboost import CatBoostClassifier

from data_loader import CSVDataLoader
from preprocess import Preprocessor
from pipeline import Pipeline
from optimisation import Optimiser


SUBMISSION_FILE = "submission.csv"


def main(input_path: Path, output_path: Path):

    model = CatBoostClassifier

    loader = CSVDataLoader(path=input_path)
    preprocessor = Preprocessor({}, {}, {})
    optimiser = Optimiser(model=model)

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
