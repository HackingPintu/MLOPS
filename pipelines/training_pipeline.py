from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import clean_data
from steps.evaluation import evaluate_model
from steps.model_train import train_model

@pipeline
def training_pipeline(data_path:str):
    df=ingest_df(data_path)
    clean_data(df)
    train_model(df)
    evaluate_model(df)