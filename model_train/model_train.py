import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb


class ModelTrainer:
    def __init__(self):
        load_dotenv()
        self.logger = logging.getLogger(__name__)


    def take_data_from_csv(self, file_name: str = "preprocessed_data.csv", folder: str = "data"
                          ) -> pd.DataFrame:
        try: 
            file_path = os.path.join(folder, file_name)
            data = pd.read_csv(file_path, index=False)
            self.logger.info(f"Preprocessed data taking from {file_path}, {len(data)} rows")
            return data

        except Exception as e:
            self.logger.exception(f"Exception raised during taking preprocessed data: {e}")
            return pd.DataFrame()
            

    def split_and_store_data(self, data: pd.DataFrame, folder: str = "data"
                            ) -> pd.DataFrame:
        try:
            X_cols = [col for col in data.columns if col != 'trip_time']
            X_train, X_validation, y_train, y_validation = train_test_split(data[X_cols], data.trip_time, test_size=0.2, random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
            self.logger.info(f"Data is splitted")

            df_train = pd.concat([X_train, y_train], axis=1)
            train_file_path = os.path.join(folder, 'train.csv')
            df_train.to_csv(train_file_path, index=False)
            self.logger.info(f"Train data stored to {train_file_path}, {len(df_train)} rows")
            
            df_test = pd.concat([X_test, y_test], axis=1)
            test_file_path = os.path.join(folder, 'test.csv')
            df_test.to_csv(test_file_path, index=False)
            self.logger.info(f"Test data stored to {test_file_path}, {len(df_test)} rows")
            
            df_validation = pd.concat([X_validation, y_validation], axis=1)
            validation_file_path = os.path.join(folder, 'validation.csv')
            df_validation.to_csv(validation_file_path, index=False)
            self.logger.info(f"Validation data stored to {validation_file_path}, {len(df_validation)} rows")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            self.logger.exception(f"Exception raised during splitting and saving data: {e}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            

    def run(
        self,
        file_name: str = "preprocessed_data.csv",
        folder: str = "data",
        model_folder: str = ".",
        model_name: str = "model.json"
    ) -> None:
        preprocessed_data = self.take_data_from_csv(file_name, folder)
        X_train, X_test, y_train, y_test = self.split_and_store_data(preprocessed_data, folder)
        
        xg_reg = None
        try:
            xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.8, learning_rate = 0.1,
                              max_depth = 10, subsample = 0.8, n_estimators = 300)
            self.logger.info(f"Model is Initialized")
    
            xg_reg.fit(X_train, y_train)
            self.logger.info(f"Model is fitted")
    
            y_pred = xg_reg.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            self.logger.info(f"RMSE: {rmse.item()}")
        except Exception as e:
            self.logger.exception(f"Exception raised during trainning model: {e}")
        
        if xg_reg is not None: 
            if not os.path.exists(model_folder):
                os.makedirs(model_folder)
            model_file_path = os.path.join(model_folder, model_name)
            xg_reg.save_model(model_file_path)
            self.logger.info(f"Model stored to {model_file_path}")