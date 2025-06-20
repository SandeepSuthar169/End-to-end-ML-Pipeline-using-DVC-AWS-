import pandas as pd
import os 
import logging
from sklearn.model_selection import train_test_split


log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

#  1.   Load data from a CSV file......
def load_data(data_url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_url)
        logger.debug('Data loaded form %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error('Faild to parse the csv file %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

#   2.   Preprocessing the data.....
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        # Correct column names (no space before colon)
        columns_to_drop = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']
        
        # Drop columns if they exist
        df.drop(columns=[col for col in columns_to_drop if col in df.columns], 
               inplace=True, 
               errors='ignore')
               
        df.rename(columns={'v1':'target', 'v2':'text'}, inplace=True)
        logger.debug('Data preprocessing completed')
        return df
    except KeyError as e:
        logger.error('Missing columns in the dataframe: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise
    


# 3. Save the train and test datasets.....
def save_data(train_data: pd.DataFrame, test_data:pd.DataFrame, data_path: str) -> None:
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(raw_data_path, 'test_data.csv'),index = False)  
        logger.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s',e)
        raise

# 4.     
def main():
    try:
        test_size = 0.2
        data_path = 'https://raw.githubusercontent.com/SandeepSuthar169/Datasets/refs/heads/main/spam.csv'
        df = load_data(data_url=data_path)
        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state= 42)
        save_data(train_data, test_data, data_path='./data')
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Erroe: {e}")

if __name__ == '__main__':
    main()            
