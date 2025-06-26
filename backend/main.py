import datetime
import uvicorn
import asyncio
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Annotated
from sqlalchemy.orm import Session
from services.Processes import Process

app = FastAPI()

origins = [
    "http://localhost:8000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/getPredictions", response_model=None)
async def get_Predictions():
    
    data = Process.Data()
    pred, lbl, img = data.get_prediction()

    returnDto = {
    "Prediction": str(pred),
    "Label": str(lbl),
    "img": img.show()
    }

    return returnDto

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)