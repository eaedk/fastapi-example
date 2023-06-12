from fastapi import FastAPI
import uvicorn
from typing import List, Literal, Optional
from pydantic import BaseModel
import pandas as pd
import pickle
import os
import json
import logging

# logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

# logging.debug('This is a debug message')
# logging.info('This is an info message')
# logging.warning('This is a warning message')
# logging.error('This is an error message')
# logging.critical('This is a critical message')


# Util Functions & Classes
def loading(fp):
    with open(fp, "rb") as f:
        data = pickle.load(f)

    print(f"INFO: Loaded data : {data}")
    return data


def predict(df, endpoint="simple"):
    """Take a dataframe as input and use it to make predictions"""

    print(
        f"[Info] 'predict' function has been called through the endpoint '{endpoint}'.\n"
    )
    
    logging.info(f" \n{df.to_markdown()}")

    # scaling
    scaled_df = scaler.transform(df)
    logging.info(f"     Scaler output is of type {type(scaled_df)}")

    # prediction
    prediction = model.predict_proba(scaled_df)
    print(f"INFO: Prediction output: {prediction}")

    # Formatting of the prediction
    ## extract highest proba
    highest_proba = prediction.max(axis=1)
    print(f"INFO: Highest probabilities : {highest_proba}")

    ## extract indexes of the highest proba
    highest_proba_idx = prediction.argmax(axis=1)
    print(f"INFO: Highest probability indexes : {highest_proba_idx}")

    ## Maching prediction with classes
    predicted_classes = [labels[i] for i in highest_proba_idx]
    print(f"INFO: Predicted classes : {predicted_classes}")
    # prediction[:, highest_proba_idx]

    # save in df
    df["predicted proba"] = highest_proba
    df["predicted label"] = predicted_classes

    print(f"INFO: dataframe filled with prediction\n{df.to_markdown()}\n")

    # parsing prediction
    # parsed = json.loads(df.to_json(orient="index")) # or
    parsed = df.to_dict("records")

    return parsed


## INPUT MODELING
class Land(BaseModel):
    """Modeling of one input data in a type-restricted dictionary-like format

    column_name : variable type # strictly respect the name in the dataframe header.

    eg.:
    =========
    customer_age : int
    gender : Literal['male', 'female', 'other']
    """

    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float


class Lands(BaseModel):
    inputs: List[Land]

    def return_list_of_dict(
        cls,
    ):
        # return [land.dict() for land in cls.inputs]
        return [i.dict() for i in cls.inputs]


# API
app = FastAPI(title="API")
ml_objects = loading(fp=os.path.join("assets", "ml", "crop_recommandation2.pkl"))
## Extract the ml components
model = ml_objects["model"]
scaler = ml_objects["scaler"].set_output(transform="pandas")
labels = ml_objects["labels"]
#  = ml_objects[""]


# Endpoints
@app.get("/")
def root():
    return {"API": " This is an API to ... ."}


@app.get("/checkup")
def test(a: Optional[int], b: int):
    return {"a": a, "b": b}


## ML endpoint
@app.post("/predict")
def make_prediction(
    N: float,
    P: float,
    K: float,
    temperature: float,
    humidity: float,
    ph: float,
    rainfall: float,
):
    """Make prediction with the passed data"""

    df = pd.DataFrame(
        {
            "N": [N],
            "P": [P],
            "K": [K],
            "temperature": [temperature],
            "humidity": [humidity],
            "ph": [ph],
            "rainfall": [rainfall],
        }
    )

    parsed = predict(df=df)  # df.to_dict('records')

    return {
        "output": parsed,
    }


@app.post("/predict_multi")
def make_multi_prediction(multi_lands: Lands):
    """Make prediction with the passed data"""
    print(f"Mutiple inputs passed: {multi_lands}\n")
    df = pd.DataFrame(multi_lands.return_list_of_dict())

    parsed = predict(df=df, endpoint="multi inputs")  # df.to_dict('records')

    return {
        "output": parsed,
        "author": "Stella Archar",
        "api_version": ";)",
    }


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)
