from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pickle

app = FastAPI(title="Spam Email Detector API")

templates = Jinja2Templates(directory="templates")

# Load trained model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


class EmailInput(BaseModel):
    email: str


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(name="index.html",request=request)


@app.post("/predict")
def predict(data: EmailInput):
    transformed_email = vectorizer.transform([data.email])
    prediction = model.predict(transformed_email)[0]

    result = "Spam" if prediction == 1 else "Not Spam"

    return JSONResponse({
        "prediction": result
    })