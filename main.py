from fastapi import FastAPI
import uvicorn
from typing import List, Literal, Optional
from pydantic import BaseModel
import pandas as pd
import pickle
import os
import json
import logging

logger = logging.getLogger(__name__)

# logging.debug('This is a debug message')
# logging.info('This is an info message')
# logging.warning('This is a warning message')
# logging.error('This is an error message')
# logging.critical('This is a critical message')

# Util Functions
def loading(fp):
    with open(fp, "rb") as f:
        data = pickle.load(f)

    print(f"INFO: Loaded data : {data}")
    return data


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
def test(a:Optional[int], b:int):
    return {"a":a, "b":b}

## ML endpoint
@app.post("/predict")
def make_prediction(N:float, P:float, K:float, temperature:float, humidity:float, ph:float, rainfall:float):
    """Make prediction with the passed data
    """
    
    df = pd.DataFrame(
        {
            "N":[N], "P":[P], "K":[K], "temperature":[temperature], "humidity":[humidity], "ph":[ph], "rainfall":[rainfall]
        }
    )
    

    print(f"{df.to_markdown()}\n")
    logger.info(f"{df.to_markdown()}")

    # scaling 
    scaled_df = scaler.transform(df)
    print(f"INFO: Scaler output is of type {type(scaled_df)}")

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
    predicted_classes = [ labels[i] for i in highest_proba_idx ]
    print(f"INFO: Predicted classes : {predicted_classes}")
    # prediction[:, highest_proba_idx]

    #save in df 
    df["predicted proba"] = highest_proba
    df["predicted label"] = predicted_classes

    print(f"INFO: dataframe filled with prediction\n{df.to_markdown()}\n")

    #parsing prediction
    # parsed = json.loads(df.to_json(orient="index")) # or
    parsed = df.to_dict('records')

    return {
        "output": parsed,
    }


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)