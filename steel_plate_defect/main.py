from pathlib import Path
import argparse

import pandas as pd
from catboost import CatBoostClassifier

from data_loader import CSVDataLoader
from preprocess import Preprocessor
from pipeline import Pipeline


SUBMISSION_FILE = 'submission.csv'


def main(input_path: Path, output_path: Path):

    loader = CSVDataLoader(path=input_path)
    model = CatBoostClassifier()
    preprocessor = Preprocessor({}, {}, {})

    pipeline = Pipeline(model=model, preprocessor=preprocessor, data_loader=loader, output_path=output_path)

    pipeline.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, required=True)
    parser.add_argument('-o', type=str, required=False, default=SUBMISSION_FILE)
    args = parser.parse_args()

    main(Path(args.i), Path(args.o))
