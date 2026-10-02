"""Create an ML-ready CSV from scikit-learn's Wisconsin diagnostic dataset."""

from pathlib import Path

from sklearn.datasets import load_breast_cancer


def main() -> None:
    dataset = load_breast_cancer(as_frame=True)
    frame = dataset.frame.rename(columns={"target": "diagnosis"})
    frame.columns = [name.strip().replace(" ", "_") for name in frame.columns]
    frame["diagnosis"] = frame["diagnosis"].map(
        {index: name for index, name in enumerate(dataset.target_names)}
    )
    destination = Path(__file__).with_name("reference_data") / "breast_cancer_diagnostic.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    print(f"Created {destination} with {len(frame):,} rows and {len(frame.columns)} columns")


if __name__ == "__main__":
    main()
