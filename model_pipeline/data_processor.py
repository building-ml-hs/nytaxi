import pandas as pd
import logging
import os


class DataProcessor:
    def __init__(
        self, data_filename: str, zones_filename: str, output_folder: str
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.data_filename = data_filename
        self.zones_filename = zones_filename
        self.output_folder = output_folder

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_filename)
        df_zone = pd.read_csv(self.zones_filename)
        return df, df_zone

    def extract_features(
        self, df: pd.DataFrame, remove_invalid: bool = True, keep_trip_id: bool = False
    ) -> pd.DataFrame:
        logging.info("Start extracting features")

        df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
        df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])

        df = df.sort_values("tpep_pickup_datetime").reset_index(drop=True)

        df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
        df["pickup_minute"] = df["tpep_pickup_datetime"].dt.minute
        df["pickup_dayofweek"] = df["tpep_pickup_datetime"].dt.dayofweek
        df["pickup_dayofmonth"] = df["tpep_pickup_datetime"].dt.day

        df["trip_time"] = (
            df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
        ).apply(lambda x: x.total_seconds() / 60)

        # Remove rows with invalid values
        if remove_invalid:
            df = df[
                (df["trip_distance"] > 0)
                & (df["passenger_count"] > 0)
                & (df["fare_amount"] > 0)
                & (df["total_amount"] > 0)
                & (df["trip_time"] > 0)
                & (df["trip_time"] < 300)
            ]

        df["is_from_airport"] = df["Airport_fee"].apply(lambda x: 1 if x > 0 else 0)

        df = df.drop(
            columns=[
                "VendorID",
                "tpep_pickup_datetime",
                "tpep_dropoff_datetime",
                "passenger_count",
                "RatecodeID",
                "store_and_fwd_flag",
                "payment_type",
                "fare_amount",
                "extra",
                "mta_tax",
                "tip_amount",
                "tolls_amount",
                "total_amount",
                "improvement_surcharge",
                "congestion_surcharge",
                "Airport_fee",
            ]
        )
        if not keep_trip_id:
            df = df.drop(columns=["trip_id"])

        return df

    def merge_location_data(
        self, df: pd.DataFrame, df_zone: pd.DataFrame
    ) -> pd.DataFrame:
        logging.info("Start merging location data")
        df = pd.merge(
            df, df_zone, left_on="PULocationID", right_on="LocationID", how="left"
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
            df, df_zone, left_on="DOLocationID", right_on="LocationID", how="left"
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

    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Start encoding categorical data")
        df = pd.get_dummies(
            df,
            columns=[
                "pickup_borough",
                "pickup_service_zone",
                "dropoff_borough",
                "dropoff_service_zone",
            ],
            drop_first=True,
        )
        return df

    def split_and_save_data(
        self,
        df: pd.DataFrame,
        train_filename: str,
        val_filename: str,
        test_filename: str,
    ) -> None:
        logging.info("Start splitting and saving data")
        total_rows = len(df)
        train_end = int(total_rows * 0.7)
        val_end = int(total_rows * 0.85)

        train = df.iloc[:train_end]
        val = df.iloc[train_end:val_end]
        test = df.iloc[val_end:]

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        train_filepath = os.path.join(self.output_folder, train_filename)
        val_filepath = os.path.join(self.output_folder, val_filename)
        test_filepath = os.path.join(self.output_folder, test_filename)

        train.to_csv(train_filepath, index=False)
        val.to_csv(val_filepath, index=False)
        test.to_csv(test_filepath, index=False)

        self.logger.info(
            f"Data split and saved: train - {len(train)}, val - {len(val)}, test - {len(test)}"
        )

    def run(self):
        df, df_zone = self.load_data()
        df = self.extract_features(df)
        df = self.merge_location_data(df, df_zone)
        df = self.encode_categorical(df)
        df = df.astype(float)
        self.split_and_save_data(df, "train.csv", "val.csv", "test.csv")
