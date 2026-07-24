"""Pelatihan model untuk MLflow Project pada workflow CI (kriteria 3).

Berbeda dengan kriteria 2, skrip ini mencatat ke mlruns/ lokal milik runner,
bukan ke DagsHub. `mlflow build-docker` membutuhkan model URI lokal, dan
menggantungkan artefak CI pada token eksternal membuat build gagal ketika
jaringan bermasalah.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

TARGET = "Churn"
RANDOM_STATE = 42


def load_datasets(data_dir: Path):
    """Muat dataset hasil preprocessing yang sudah siap dilatih."""
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    return (
        train.drop(columns=[TARGET]),
        train[TARGET],
        test.drop(columns=[TARGET]),
        test[TARGET],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="telco_preprocessing")
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=20)
    args = parser.parse_args()

    data_dir = Path(__file__).resolve().parent / args.data_dir
    x_train, y_train, x_test, y_test = load_datasets(data_dir)
    print(f"[INFO] data latih: {x_train.shape}, data uji: {x_test.shape}")

    with mlflow.start_run(run_name="ci_random_forest") as run:
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("random_state", RANDOM_STATE)

        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)

        predictions = model.predict(x_test)
        probabilities = model.predict_proba(x_test)[:, 1]
        for name, value in {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions),
            "recall": recall_score(y_test, predictions),
            "f1_score": f1_score(y_test, predictions),
            "roc_auc": roc_auc_score(y_test, probabilities),
        }.items():
            mlflow.log_metric(name, value)
            print(f"[HASIL] {name}: {value:.4f}")

        # Signature dideklarasikan sebagai float64 seluruhnya. Kolom hasil one-hot
        # bertipe int64, dan schema enforcement MLflow menolak float yang dikirim
        # ke kolom bertipe integer - hal yang pasti terjadi saat serving karena
        # klien JSON mengirim angka tanpa membedakan int dan float.
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            input_example=x_train.head(5).astype("float64"),
        )

        # Baris terakhir dibaca oleh workflow CI untuk membangun Docker image.
        print(f"RUN_ID={run.info.run_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
