from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.schema import CustomerData
from app.predictor import predict_customer

app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0"
)

# Static Files
app.mount("/static", StaticFiles(directory="static"), name="static")

# HTML Templates
templates = Jinja2Templates(directory="templates")


# Home Page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


# Prediction API
@app.post("/predict")
def predict(data: CustomerData):
    return predict_customer(data)

    