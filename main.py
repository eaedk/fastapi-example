from fastapi import FastAPI
import uvicorn
from typing import List, Literal, Optional
from pydantic import BaseModel
import pandas as pd
import pickle
import os

# API
app = FastAPI(title="API")


@app.get("/")
def root():
    return {"API": " This is an API to ... ."}

@app.get("/checkup")
def test(a:Optional[int], b:int):
    return {"a":a, "b":b}

if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)