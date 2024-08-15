import os
import boto3
import json
import gzip
import pandas as pd
from io import BytesIO
from dotenv import load_dotenv


class LogReader:
    def __init__(self):
        load_dotenv()

        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("LOG_AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("LOG_AWS_SECRET_ACCESS_KEY"),
            endpoint_url=os.getenv("LOG_AWS_ENDPOINT_URL_S3"),
            region_name=os.getenv("LOG_AWS_REGION"),
        )

        self.bucket_name = os.getenv("LOG_BUCKET_NAME")
        self.app_name = os.getenv("APP_NAME")

    def download_logs(self, last_n_minutes: int) -> pd.DataFrame:
        """
        Download logs from S3, extract, parse JSON, and return as a DataFrame.
        :param app_name: The application name used in the S3 folder structure.
        :param last_n_minutes: The number of minutes to look back for logs.
        :return: A pandas DataFrame containing extracted features and predictions.
        """
        date_today = pd.Timestamp.now().strftime("%Y-%m-%d")
        date_yesterday = (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime(
            "%Y-%m-%d"
        )
        folder_paths = [
            f"{self.app_name}/{date_today}/",
            f"{self.app_name}/{date_yesterday}/",
        ]
        objects_today = self.s3_client.list_objects_v2(
            Bucket=self.bucket_name, Prefix=folder_paths[0]
        )
        objects_yesterday = self.s3_client.list_objects_v2(
            Bucket=self.bucket_name, Prefix=folder_paths[1]
        )

        if "Contents" not in objects_today and "Contents" not in objects_yesterday:
            return pd.DataFrame()

        all_data = []

        for obj in objects_today.get("Contents", []) + objects_yesterday.get(
            "Contents", []
        ):
            log_file_key = obj["Key"]

            # Ignore log objects that are more than last_n_minutes + 30 minutes old
            if pd.Timestamp.utcnow() - pd.Timestamp(obj["LastModified"]) > pd.Timedelta(
                minutes=last_n_minutes + 30
            ):
                continue

            log_obj = self.s3_client.get_object(
                Bucket=self.bucket_name, Key=log_file_key
            )
            with gzip.GzipFile(fileobj=BytesIO(log_obj["Body"].read())) as gz:
                for line in gz:
                    log_entry = json.loads(line.decode("utf-8"))

                    # Ignore log entries that are more than last_n_minutes old
                    if pd.Timestamp.utcnow() - pd.Timestamp(
                        log_entry["timestamp"]
                    ) > pd.Timedelta(minutes=last_n_minutes):
                        continue

                    if log_entry["message"].startswith(
                        "INFO:backend.utils:prediction_result: "
                    ):
                        log_entry["message"] = log_entry["message"].replace(
                            "INFO:backend.utils:prediction_result: ", ""
                        )
                        log_data = json.loads(log_entry["message"])
                        all_data.append(self.extract_features(log_data))

        return pd.DataFrame(all_data)

    @staticmethod
    def extract_features(log_entry: dict) -> dict:
        """
        Extract the features and prediction from a single log entry.
        :param log_entry: A dictionary representing a single log entry.
        :return: A dictionary of extracted data.
        """
        data = log_entry["extracted_features"]
        data["prediction"] = log_entry["prediction"]
        data["trip_id"] = log_entry["trip_id"]
        return data


if __name__ == "__main__":
    log_reader = LogReader()
    df = log_reader.download_logs(last_n_minutes=30)
    print(df.head())
    print(len(df))
