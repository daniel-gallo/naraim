from typing import List
from pathlib import Path

import pandas as pd
from tensorboard.backend.event_processing import event_accumulator


def get_df(folders: List[str], metrics: List[str]) -> pd.DataFrame:
    """
    Inputs:
      - folders: sorted list of tensorboard folders
      - metrics: name of the metrics that will be extracted from the tensroboard folders
    Output: dataframe with the following columns:
      - folder: name of the folder that originated this datapoint
      - step
      - metric_name
      - metric_value

      The rows are unique conditioning on (step, metric_name). In case of overlapping jobs,
      the values from the last run are taken.
    """
    df = {"folder": [], "step": [], "metric_name": [], "metric_value": []}

    for folder in folders:
        ea = event_accumulator.EventAccumulator(folder)
        ea.Reload()

        print("Read", folder)
        print("Available metrics:", ea.Tags()["scalars"])

        for metric in metrics:
            events = ea.Scalars(metric)
            df["folder"].extend([folder] * len(events))
            df["step"].extend([event.step for event in events])
            df["metric_name"].extend([metric] * len(events))
            df["metric_value"].extend([event.value for event in events])

    df = (
        pd.DataFrame(df)
        .sort_values("folder")
        .groupby(["step", "metric_name"])
        .first()
        .reset_index()
    )

    return df


if __name__ == "__main__":
    folders = [
        "../backup/6890191/tensorboard",
        "../backup/6897292/tensorboard",
        "../backup/6944198/tensorboard",
    ]
    assert folders == sorted(folders)

    metrics = ["loss/train", "loss/val"]

    df = get_df(folders, metrics)

    filename = Path("data/baseline.csv")
    filename.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filename, index=False)
