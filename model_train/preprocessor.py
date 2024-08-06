import pandas as pd

class PreProcessor:
    def __init__(self):
        load_dotenv()
        self.logger = logging.getLogger(__name__)


    def take_data_from_csv(self, file_name: str = "data.csv", folder: str = "data"
                          ) -> pd.DataFrame:
        try: 
            file_path = os.path.join(folder, file_name)
            data = pd.read_csv(file_path, index=False)
            self.logger.info(f"Trip data taking from {file_path}, {len(data)} rows")
            return data

        except Exception as e:
            self.logger.exception(f"Exception raised during taking trip data: {e}")
            return pd.DataFrame()
            

    def take_zone_data_from_csv(self, file_name: str = "zones.csv", folder: str = "data"
                          ) -> pd.DataFrame:
        try:
            file_path = os.path.join(folder, file_name)
            data = pd.read_csv(file_path, index=False)
            self.logger.info(f"Zones data taking from {file_path}, {len(data)} rows")
            return data
            
        except Exception as e:
            self.logger.exception(f"Exception raised during taking zones data: {e}")
            return pd.DataFrame()

        
    def preprocess(self, trip_data: pd.DataFrame, zones_data: pd.DataFrame) -> pd.DataFrame:
        cols = ["tpep_pickup_datetime", "tpep_dropoff_datetime", "trip_distance", "PULocationID", "DOLocationID", "store_and_fwd_flag", "Airport_fee"]
        
        try:
            trip_data = trip_data[cols]
            trip_data['pickup_hour'] = trip_data['tpep_pickup_datetime'].apply(lambda x: x.hour)
            trip_data['pickup_minute'] = trip_data['tpep_pickup_datetime'].apply(lambda x: x.minute)
            trip_data['pickup_dayofweek'] = trip_data['tpep_pickup_datetime'].apply(lambda x: x.weekday())
            trip_data['pickup_dayofmonth'] = trip_data['tpep_pickup_datetime'].apply(lambda x: x.day)
            trip_data['trip_time'] = (trip_data['tpep_dropoff_datetime'] - trip_data['tpep_pickup_datetime']) \
                                        .apply(lambda x: round(x.total_seconds()/60, 2))
            trip_data['Airport_fee'] = trip_data['Airport_fee'].replace({0.00:'0', 1.75:'1', -1.75:'2'})
            trip_data['store_and_fwd_flag'] = trip_data['store_and_fwd_flag'].replace({'N':0, 'Y':1})
            self.logger.info(f"New features are created for trip data")

            df = pd.merge(trip_data, zones_data, left_on='PULocationID', right_on='LocationID', how='left')
            df = pd.merge(df, zones_data, left_on='DOLocationID', right_on='LocationID', how='left')
            self.logger.info(f"Merge dataset is created")

            df = df.rename(columns={'Borough_x': 'pickup_Borough', 'Zone_x':'pickup_Zone', 'service_zone_x':'pickup_service_zone', \
                        'Borough_y': 'dropout_Borough', 'Zone_y':'dropout_Zone', 'service_zone_y':'dropout_service_zone'})

            df = df.drop(['LocationID_x', 'LocationID_y', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'PULocationID', 'DOLocationID', \
                          'pickup_Zone', 'dropout_Zone'], axis=1)

            df = pd.get_dummies(df, columns = ['pickup_Borough', 'pickup_service_zone', 'dropout_Borough', 'dropout_service_zone', 'Airport_fee'], \
                                drop_first=True)

            df = df.astype(float)
            self.logger.info(f"One hot encoding are implemented")
            self.logger.info(f"Preprocess is done")
            return df
                
        except Exception as e:
            self.logger.exception(f"Exception raised during preprocessing data: {e}")
            return pd.DataFrame()

    def store_data_to_csv(
        self, data: pd.DataFrame, file_name: str = "preprocessed_data.csv", folder: str = "data"
    ) -> None:
        if not os.path.exists(folder):
            os.makedirs(folder)
        file_path = os.path.join(folder, file_name)
        data.to_csv(file_path, index=False)
        self.logger.info(f"Preprocessed data stored to {file_path}, {len(data)} rows")


    def run(
        self,
        file_name: str = "data.csv",
        folder: str = "data",
        zones_file_name: str = "zones.csv",
        preprocessed_file_name: str = "preprocessed_data.csv",
    ) -> None:
        trip_data = self.take_data_from_csv(file_name, folder)
        zones_data = self.take_zone_data_from_csv(zones_file_name, folder)

        if (not trip_data.empty) and (not zones_data.empty):
            df = self.preprocess(trip_data, zones_data)
        else:
            self.logger.warning(f"Trip_data empty: {trip_data.empty}, Zones_data empty: {zones_data.empty}")

        if not df.empty:
            self.store_data_to_csv(df, preprocessed_file_name, folder)
        else:
            self.logger.warning("No preprocessed data to store")