import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay, classification_report, f1_score, precision_score,
    recall_score, roc_auc_score, root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


ROOT = Path(__file__).resolve().parents[1]


def main():
    data = pd.read_csv(
        ROOT / 'data/processed/nyc_taxi_clean.csv',
        parse_dates=['pickup_datetime'],
    )
    data['pickup_hour'] = data['pickup_datetime'].dt.hour
    data['pickup_weekday'] = data['pickup_datetime'].dt.day_of_week
    features = [
        'vendor_id', 'passenger_count', 'pickup_longitude',
        'pickup_latitude', 'pickup_hour', 'pickup_weekday',
    ]
    target = (data['trip_duration'] > 3600).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        data[features], target, test_size=0.2, random_state=42,
        stratify=target,
    )
    model = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        RandomForestClassifier(
            n_estimators=80, max_depth=12, min_samples_leaf=20,
            class_weight={0: 1, 1: 5}, random_state=42, n_jobs=-1,
        ),
    )

    mlflow.set_tracking_uri('sqlite:///' + (ROOT / 'mlflow.db').as_posix())
    mlflow.set_experiment('NYC Taxi Lab 4')
    mlflow.sklearn.autolog(log_input_examples=True, silent=True)

    with mlflow.start_run(run_name='random_forest_baseline') as run:
        mlflow.log_params({
            'test_size': 0.2, 'split_random_state': 42,
            'target_threshold_seconds': 3600,
            'features': ', '.join(features),
            'train_rows': len(X_train), 'test_rows': len(X_test),
        })
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        probability = model.predict_proba(X_test)[:, 1]
        metrics = {
            'test_roc_auc': roc_auc_score(y_test, probability),
            'test_f1': f1_score(y_test, prediction, zero_division=0),
            'test_precision': precision_score(y_test, prediction, zero_division=0),
            'test_recall': recall_score(y_test, prediction, zero_division=0),
            'test_rmse': root_mean_squared_error(y_test, prediction),
        }
        mlflow.log_metrics(metrics)
        report = classification_report(
            y_test, prediction, target_names=['обычная', 'аномальная'],
            zero_division=0,
        )
        results = ROOT / 'results'
        results.mkdir(exist_ok=True)
        (results / 'metrics.json').write_text(
            json.dumps(metrics, indent=2), encoding='utf-8',
        )
        (results / 'report.txt').write_text(report, encoding='utf-8')
        ConfusionMatrixDisplay.from_predictions(
            y_test, prediction, display_labels=['обычная', 'долгая'],
        )
        plt.title('Результат на тестовой выборке')
        plt.savefig(results / 'test_confusion_matrix.png', bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(str(results / 'test_confusion_matrix.png'))
        mlflow.log_artifact(str(results / 'metrics.json'))
        mlflow.log_artifact(str(results / 'report.txt'))
        print(report)
        print(json.dumps(metrics, indent=2))
        print('MLflow run:', run.info.run_id)


if __name__ == '__main__':
    main()
