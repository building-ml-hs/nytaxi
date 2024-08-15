import os
import prefect
from dotenv import load_dotenv
import pandas as pd
from typing import List
from evidently.report import Report
from evidently.metric_preset import RegressionPreset, TargetDriftPreset
from evidently.ui.workspace.cloud import CloudWorkspace

import model_pipeline.data_collector
import model_pipeline.data_processor
import monitoring.log_reader


@prefect.task
def get_data_from_db() -> pd.DataFrame:
    collector = model_pipeline.data_collector.DataCollector()
    processor = model_pipeline.data_processor.DataProcessor("", "", "")

    df = collector.get_data(days=2)
    zones_df = collector.collect_zones_data()

    df = processor.extract_features(df, keep_trip_id=True)
    df = processor.merge_location_data(df, zones_df)
    df = processor.encode_categorical(df)

    trip_ids = df.trip_id
    df = df.drop(columns=["trip_id"])
    df = df.astype(float)
    df["trip_id"] = trip_ids
    df["trip_id"] = df["trip_id"].astype(str)
    df = df.rename(columns={"trip_time": "target"})

    return df


@prefect.task
def get_data_from_logs() -> pd.DataFrame:
    log_reader = monitoring.log_reader.LogReader()
    df = log_reader.download_logs(last_n_minutes=60)
    df["trip_id"] = df["trip_id"].astype(str)
    return df


@prefect.task
def read_data_from_csvs(types: List[str]) -> pd.DataFrame:
    dfs = []
    for type in types:
        df = pd.read_csv(f"./data/{type}.csv")
        dfs.append(df)
    return pd.concat(dfs)


@prefect.task
def generate_metrics_report(df: pd.DataFrame) -> None:
    if len(df) == 0:
        raise ValueError("No common trip_id between logs and db data")

    report = Report(metrics=[RegressionPreset()])
    report.run(reference_data=None, current_data=df)

    return report


@prefect.task
def generate_target_drift_report(curr_df: pd.DataFrame, ref_df: pd.DataFrame) -> None:
    ref_df = ref_df.rename(columns={"trip_time": "target"})
    curr_df = curr_df.rename(columns={"trip_time": "target"})
    curr_df = curr_df.drop(columns=["prediction"])
    print("Len df", len(ref_df))
    print("Len target ", len(ref_df["target"]))

    report = Report(metrics=[TargetDriftPreset()])
    report.run(reference_data=ref_df, current_data=curr_df)

    return report


@prefect.task
def upload_reports(metrics_report: Report, target_drift_report: Report) -> None:
    load_dotenv()
    evidently_token = os.getenv("EVIDENTLY_TOKEN")
    evidently_project_id = os.getenv("EVIDENTLY_PROJECT_ID")

    ws = CloudWorkspace(token=evidently_token, url="https://app.evidently.cloud")

    project = ws.get_project(evidently_project_id)

    ws.add_report(project.id, metrics_report, "metrics_report")
    ws.add_report(project.id, target_drift_report, "target_drift_report")


def generate_flow_run_name():
    time_now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    return f"{time_now}"


@prefect.flow(
    name="NY Taxi - Model Metrics Report", flow_run_name=generate_flow_run_name
)
def model_metrics_report() -> None:
    db_df = get_data_from_db()
    logs_df = get_data_from_logs()
    train_val_df = read_data_from_csvs(["train", "val"])
    print("DB DF", db_df["trip_id"].iloc[:5])
    print("LOGS DF", logs_df["trip_id"].iloc[:5])
    merged_df = pd.merge(logs_df, db_df, on="trip_id", how="inner")
    metrics_report = generate_metrics_report(df=merged_df)
    target_drift_report = generate_target_drift_report(
        curr_df=merged_df, ref_df=train_val_df
    )
    upload_reports(
        metrics_report=metrics_report, target_drift_report=target_drift_report
    )


if __name__ == "__main__":
    model_metrics_report.serve(
        name="NY Taxi - Scheduled Model Metrics Report", cron="*/30 * * * *"
    )
