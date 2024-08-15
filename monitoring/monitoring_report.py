import pandas as pd
import os
from monitoring.log_reader import LogReader
import xgboost as xgb
import prefect
from typing import List

from evidently.ui.workspace.cloud import CloudWorkspace
from evidently.report import Report
from evidently.metric_preset import DataQualityPreset
from evidently.metric_preset import DataDriftPreset
from evidently.metric_preset import TargetDriftPreset


@prefect.task
def get_data_from_logs() -> pd.DataFrame:
    log_reader = LogReader()
    df = log_reader.download_logs(last_n_minutes=30)
    # df["trip_id"] = df["trip_id"].astype(str)
    return df


@prefect.task
def read_data_from_csvs(types: List[str]) -> pd.DataFrame:
    dfs = []
    for type in types:
        df = pd.read_csv(f"./data/{type}.csv")
        dfs.append(df)
    return pd.concat(dfs)


@prefect.task
def create_data_report(df_current: pd.DataFrame, df_ref: pd.DataFrame):

    report_cols = [
        col for col in df_current.columns if col not in ["trip_id", "prediction"]
    ]

    data_report = Report(
        metrics=[
            DataDriftPreset(stattest="psi", stattest_threshold="0.3"),
            DataQualityPreset(),
        ],
    )

    data_report.run(
        reference_data=df_ref[report_cols], current_data=df_current[report_cols]
    )

    return data_report


@prefect.task
def create_prediction_drift_report(df_current: pd.DataFrame, df_ref: pd.DataFrame):

    model = xgb.Booster()
    model.load_model("models/xgb.json")

    df_ref["prediction"] = model.predict(
        xgb.DMatrix(df_ref.drop(columns=["trip_time"]))
    )

    report_cols = [
        col for col in df_current.columns if col not in ["trip_id", "trip_time"]
    ]

    prediction_report = Report(
        metrics=[
            TargetDriftPreset(),
        ],
    )

    prediction_report.run(
        reference_data=df_ref[report_cols], current_data=df_current[report_cols]
    )

    return prediction_report


@prefect.task
def upload_reports(data_report: Report, prediction_report: Report) -> None:
    evidently_token = os.getenv("EVIDENTLY_TOKEN")
    evidently_project_id = os.getenv("EVIDENTLY_PROJECT_ID")

    ws = CloudWorkspace(token=evidently_token, url="https://app.evidently.cloud")

    project = ws.get_project(evidently_project_id)

    ws.add_report(project.id, data_report)
    ws.add_report(project.id, prediction_report)


def generate_flow_run_name():
    time_now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    return f"{time_now}"


@prefect.flow(
    name="NY Taxi - Data Drift and Quality Report", flow_run_name=generate_flow_run_name
)
def monitoring_report() -> None:
    df_current = get_data_from_logs()
    df_train_val = read_data_from_csvs(types=["train", "val"])
    df_test = read_data_from_csvs(types=["test"])
    data_report = create_data_report(df_current=df_current, df_ref=df_train_val)
    prediction_report = create_prediction_drift_report(
        df_current=df_current, df_ref=df_test
    )
    upload_reports(data_report=data_report, prediction_report=prediction_report)


if __name__ == "__main__":
    monitoring_report.serve(name="NY Taxi - Scheduled Report", cron="*/5 * * * *")
