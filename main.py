import os 
import warnings
warnings.filterwarnings('ignore')   # filtering the warnings
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn
import dagshub    # for storing the code in the cloud
import logging


# creating logger
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# creating the evaluation metrices
def eval(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2_score = r2_score(actual, pred)
    return rmse, mae, r2_score

if __name__ == '__main__':
    np.random.seed(40)

    # reading the data from the Github url NOTE: URL must direct to the raw csv file.
    csv_url = (
        "https://raw.githubusercontent.com/druvpat01/Datasets/refs/heads/main/winequality-red.csv"
        )
    
    try:
        data = pd.read_csv(csv_url, sep=',')
    except Exception as e:
        logger.exception("Unable to load data from the url. Error: ",e)
    
    # split data into training and test sets (0.75, 0.25)
    train, test = train_test_split(data, train_size=0.75)

    # predicted values is 'quality' which is scalar between [3,9]
    train_x = train.drop(['quality'], axis=1)
    test_x = test.drop(['quality'], axis=1)
    train_y = train[['quality']]
    test_y = test[['quality']]

    # taking only two hyperparameters into consideration - alpha value, l1_ratio
    #NOTE: 'sys.argv' allows to pass input at time of execution of code. eg. 'python main.py 0.3 0.6'
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 1 else 0.5

    #using mlflow
    with mlflow.start_run():
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        model.fit(train_x, train_y)

        pred = model.predict(test_x)

        rmse, mae, r2_score = eval(test_y, pred)

        print(f"ElasticNet model (alpha = {alpha}, l1_ratio = {l1_ratio})")
        print(' RMSE: ', rmse)
        print(' MAS: ',mae)
        print(' R2_score: ', r2_score)

        mlflow.log_param('alpha', alpha)
        mlflow.log_param('l1_ratio', l1_ratio)
        mlflow.log_metric('rmse', rmse)
        mlflow.log_metric('mae', mae)
        mlflow.log_metric('r2_score', r2_score)


