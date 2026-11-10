import argparse
from pathlib import Path

from sklearn.datasets import load_iris


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=["v1", "v2"], default="v1")
    args = parser.parse_args()
    df = load_iris(as_frame=True).frame
    df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "target"]
    # v2 changes row order only: a genuine content version, without inventing labels.
    if args.version == "v2":
        df = df.sample(frac=1, random_state=17).reset_index(drop=True)
    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/iris.csv", index=False, float_format="%.1f")
    print(f"Prepared {args.version}: {len(df)} rows from sklearn.datasets.load_iris")


if __name__ == "__main__":
    main()
