import os
import logging
import pandas as pd
from backend.api_models import PredictionRequest


class FeatureExtractor:
    def __init__(
        self, zones_filename: str, data_folder: str, test_filename: str
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.zones_filename = zones_filename
        self.folder = data_folder
        self.zones_filepath = os.path.join(data_folder, zones_filename)
        self.df_zone = pd.read_csv(self.zones_filepath)
        self.test_filepath = os.path.join(data_folder, test_filename)

    def _get_dummy_columns(self):
        test_df_top = pd.read_csv(self.test_filepath, nrows=0)
        test_df_top.drop(columns=["trip_time"], inplace=True)
        return test_df_top.columns.tolist()

    def _reorder_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        expected_columns = self._get_dummy_columns()

        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0

        return df[expected_columns]

    def _merge_zones(self, df: pd.DataFrame) -> pd.DataFrame:
        df = pd.merge(
            df, self.df_zone, left_on="PULocationID", right_on="LocationID", how="left"
        )

        unmatched_rows = df[df["LocationID"].isnull()]
        if not unmatched_rows.empty:
            self.logger.error(
                f"{len(unmatched_rows)} pickup location ID not found, example ID from request: {unmatched_rows['PULocationID'].values[0]}"
            )

        df = df.rename(
            columns={
                "Borough": "pickup_borough",
                "Zone": "pickup_zone",
                "service_zone": "pickup_service_zone",
            }
        )
        df = df.drop(columns=["PULocationID", "pickup_zone", "LocationID"])

        df = pd.merge(
            df, self.df_zone, left_on="DOLocationID", right_on="LocationID", how="left"
        )

        unmatched_rows = df[df["LocationID"].isnull()]
        if not unmatched_rows.empty:
            self.logger.error(
                f"{len(unmatched_rows)} dropoff location ID not found, example ID from request: {unmatched_rows['DOLocationID'].values[0]}"
            )

        df = df.rename(
            columns={
                "Borough": "dropoff_borough",
                "Zone": "dropoff_zone",
                "service_zone": "dropoff_service_zone",
            }
        )
        df = df.drop(columns=["DOLocationID", "dropoff_zone", "LocationID"])

        return df

    def _one_hot_encode(self, df: pd.DataFrame) -> pd.DataFrame:

        # Manually create one-hot encoding for categorical columns
        pickup_borough = f"pickup_borough_{df.at[0, 'pickup_borough']}"
        pickup_service_zone = f"pickup_service_zone_{df.at[0, 'pickup_service_zone']}"
        dropoff_borough = f"dropoff_borough_{df.at[0, 'dropoff_borough']}"
        dropoff_service_zone = (
            f"dropoff_service_zone_{df.at[0, 'dropoff_service_zone']}"
        )

        # Initialize the dummy columns with zeros
        for column in self._get_dummy_columns():
            if column not in df.columns:
                df[column] = 0

        # Set the respective column to 1
        if pickup_borough in df.columns:
            df[pickup_borough] = 1
        else:
            self.logger.error(f"Column {pickup_borough} not found in the dataframe")

        if pickup_service_zone in df.columns:
            df[pickup_service_zone] = 1

        if dropoff_borough in df.columns:
            df[dropoff_borough] = 1

        if dropoff_service_zone in df.columns:
            df[dropoff_service_zone] = 1

        return df

    def extract_features(self, request: PredictionRequest) -> pd.DataFrame:

        expected_format = "%Y-%m-%dT%H/%M/%S%z"
        try:
            request_datetime = pd.to_datetime(
                request.request_datetime, format=expected_format
            )
        except ValueError:
            try:
                request_datetime = pd.to_datetime(request.request_datetime)
            except ValueError:
                self.logger.error(
                    f"Request datetime {request.request_datetime} does not match the expected format {expected_format} nor the default format"
                )
                raise

        if request.trip_distance < 0:
            self.logger.warning(
                f"Trip distance is negative: {request.trip_distance}, converting to positive"
            )

        features = {
            "trip_distance": abs(request.trip_distance),
            "pickup_hour": request_datetime.hour,
            "pickup_minute": request_datetime.minute,
            "pickup_dayofweek": request_datetime.dayofweek,
            "pickup_dayofmonth": request_datetime.day,
            "is_from_airport": request.Airport,
            "PULocationID": request.PULocationID,
            "DOLocationID": request.DOLocationID,
        }

        df = pd.DataFrame([features])
        df = self._merge_zones(df)
        df = self._one_hot_encode(df)
        df = self._reorder_columns(df)
        df = df.astype(float)
        return df
