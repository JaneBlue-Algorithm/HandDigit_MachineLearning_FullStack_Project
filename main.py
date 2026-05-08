from fastapi import FastAPI
from pydantic import BaseModel, conlist
import numpy as np
import pickle
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔥 SAFE MODEL LOAD
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")

model = pickle.load(open(model_path, "rb"))

# 🔥 FIXED INPUT (16 feature zorunlu)
class InputData(BaseModel):
    features: conlist(float, min_length=16, max_length=16)

@app.post("/predict")
def predict(data: InputData):
    arr = np.array(data.features).reshape(1, -1)

    prediction = model.predict(arr)

    return {
        "prediction": int(prediction[0])
    }









# from fastapi import FastAPI
# from pydantic import BaseModel
# import numpy as np
# import pickle
# from fastapi.middleware.cors import CORSMiddleware

# app = FastAPI()


# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # 🔥 test için hepsine izin
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# # model yükle
# model = pickle.load(open("model.pkl", "rb"))

# class InputData(BaseModel):
#     features: list

# @app.post("/predict")
# def predict(data: InputData):
#     arr = np.array(data.features).reshape(1, -1)

#     prediction = model.predict(arr)

#     return {"prediction": int(prediction[0])}










# conda activate handdigit 
# uvicorn main:app --reload

# http://127.0.0.1:8000/docs tarayicida calisiyor 
#swagger da bu kodu gonder 8 ciktisi gelir 
# {
#   "features": [47,100,27,81,57,37,26,0,0,23,56,53,100,90,40,98]
# }


# npx create-react-app@latest handdigit-frontend
# cd handdigit-frontend
# npm start
# http://localhost:3000/ bu linkte npm aciliyor 
