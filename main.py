import dill as dill
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from apscheduler.schedulers.blocking import BlockingScheduler
from tzlocal import get_localzone

app = FastAPI()
df = pd.read_csv('model/data/merged_data_final.csv').drop('flag', axis=1)
with open('model/car_pipe_final.pkl', 'rb') as file:
    loaded_data = dill.load(file)
    model = loaded_data['model']
    preprocessor = loaded_data['preprocessor']
    columns_expected = loaded_data['columns']
    categorical_features_indices_saved = loaded_data['categorical_features_indices']

sched = BlockingScheduler(timezone=get_localzone())

if preprocessor is None:
    print("Preprocessor not found in the loaded data.")
processed_data = preprocessor.transform(df)

class Form(BaseModel):
    session_id: str
    client_id: float
    visit_date: str
    visit_time: str
    visit_number: int
    utm_source: str
    utm_medium: str
    utm_campaign: str
    utm_adcontent: str
    utm_keyword: str
    device_category: str
    device_os: str
    device_brand: str
    device_model: str
    device_screen_resolution: str
    device_browser: str
    geo_country: str
    geo_city: str


class Prediction(BaseModel):
    client_id: str
    action: float

@app.get('/status')
def status():
    return "Car Action Prediction Service"

@app.get('/version')
def version():
    if 'metadata' in loaded_data:
        return loaded_data['metadata']
    else:
        return JSONResponse(status_code=404, content={"message": "Metadata not found"})

@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    print("Received form data:", form.dict())
    input_df = pd.DataFrame([form.dict()])
    processed_data = preprocessor.transform(input_df)
    prediction = model.predict(processed_data)[0]
    return {
        'client_id': str(form.client_id),
        'action': float(prediction)  # Убедитесь, что prediction действительно float
    }

@sched.scheduled_job('interval', seconds=480)
def on_time():
    try:
        processed_data = preprocessor.transform(df)
        data = processed_data.sample(n=500)
        data['predicted_action'] = model.predict(data)
        print(data[['client_id', 'predicted_action']])
    except Exception as e:
        print("Error during job execution:", e)

if __name__ == '__main__':
    sched.start()