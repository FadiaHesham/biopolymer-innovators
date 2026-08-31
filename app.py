from fastapi import FastAPI, Form 
from pydantic import BaseModel, Field 
import pandas as pd 
import joblib 
 
 
# ============================================================ 
# LOAD MODELS 
# ============================================================ 
 
yield_model = joblib.load("PHA_yield_model.pkl") 
 
digital_twin_package = joblib.load("digital_twin_model_v2.pkl") 
 
digital_twin_model = digital_twin_package["model"] 
digital_twin_features = digital_twin_package["features"] 
digital_twin_targets = digital_twin_package["targets"] 
 
 
# ============================================================ 
# FASTAPI APP 
# ============================================================ 
 
app = FastAPI( 
    title="BioPolymer Innovators - PHA AI System", 
    description="AI Pipeline: PHA Yield Prediction + Digital Twin Forecast", 
    version="2.0" 
) 
 
 
# ============================================================ 
# USER INPUT - JSON 
# ============================================================ 
 
class PHAInput(BaseModel): 
 
    Rice_Straw_Mass: float = Field( 
        ..., 
        gt=0, 
        description="Rice straw mass in kg. Example: 1.5" 
    ) 
 
    Moisture: float = Field( 
        ..., 
        ge=0, 
        le=100, 
        description="Moisture percentage. Example: 10.5" 
    ) 
 
    Initial_Sugar: float = Field( 
        ..., 
        gt=0, 
        description="Initial sugar concentration in g/L. Example: 25" 
    ) 
 
    Temperature: float = Field( 
        ..., 
        description="Fermentation temperature in °C. Example: 30" 
    ) 
 
    pH: float = Field( 
        ..., 
        description="Fermentation pH. Example: 6.8" 
    ) 
 
    DO: float = Field( 
        ..., 
        ge=0, 
        le=100, 
        description="Dissolved oxygen percentage. Example: 35" 
    ) 
 
    RPM: float = Field( 
        ..., 
        ge=0, 
        description="Agitation speed in RPM. Example: 180" 
    ) 
 
    Fermentation_Time: float = Field( 
        ..., 
        ge=0, 
        description="Fermentation time in hours. Example: 20" 
    ) 
 
 
# ============================================================ 
# HOME 
# ============================================================ 
 
@app.get("/") 
def home(): 
 
    return { 
        "message": "BioPolymer Innovators PHA AI System is running", 
 
        "models": [ 
            "Model 1 - PHA Yield Prediction", 
            "Model 2 - Digital Twin Forecast" 
        ] 
    } 
 
 
# ============================================================ 
# CORE PREDICTION FUNCTION 
# ============================================================ 
 
def run_prediction( 
    Rice_Straw_Mass, 
    Moisture, 
    Initial_Sugar, 
    Temperature, 
    pH, 
    DO, 
    RPM, 
    Fermentation_Time 
): 
 
    # ======================================================== 
    # MODEL 1 
    # PHA YIELD PREDICTION 
    # ======================================================== 
 
    model1_input = pd.DataFrame([{ 
 
        "Rice_Straw_Mass": Rice_Straw_Mass, 
 
        "Moisture": Moisture, 
 
        "Initial_Sugar": Initial_Sugar, 
 
        "Temperature": Temperature, 
 
        "pH": pH, 
 
        "DO": DO, 
 
        "RPM": RPM, 
 
        "Fermentation_Time": Fermentation_Time 
 
    }]) 
 
 
    predicted_yield = float( 
        yield_model.predict(model1_input)[0] 
    ) 
 
 
    # ======================================================== 
    # CURRENT FERMENTATION STATE 
    # ======================================================== 
 
    time_hours = Fermentation_Time 
 
 
    # Temporary estimated values 
    # These will later be replaced by real sensor readings. 
 
    sugar_concentration = max( 
        0, 
        Initial_Sugar * (1 - 0.025 * time_hours) 
    ) 
 
 
    biomass = max( 
        0.8, 
        min( 
            4.8, 
            0.8 + (time_hours / 40) * 3.5 
        ) 
    ) 
 
 
    pha_accumulation = max( 
        5, 
        min( 
            60, 
            5 + (time_hours / 40) * 50 
        ) 
    ) 
 
 
    # ======================================================== 
    # MODEL 2 INPUT 
    # ======================================================== 
    # Predicted PHA Yield comes automatically from Model 1. 
 
    model2_input = pd.DataFrame([{ 
 
        "Time_Hours": time_hours, 
 
        "Temperature": Temperature, 
 
        "pH": pH, 
 
        "DO": DO, 
 
        "RPM": RPM, 
 
        "Sugar_Concentration": sugar_concentration, 
 
        "Biomass": biomass, 
 
        "PHA_Accumulation": pha_accumulation, 
 
        "Predicted_PHA_Yield": predicted_yield 
 
    }]) 
 
 
    # ======================================================== 
    # DIGITAL TWIN 
    # Forecast next 5 hours 
    # ======================================================== 
 
    twin_prediction = digital_twin_model.predict( 
        model2_input[digital_twin_features] 
    )[0] 
 
 
    forecast = dict( 
        zip( 
            digital_twin_targets, 
            twin_prediction 
        ) 
    ) 
 
 
    # ======================================================== 
    # FUTURE CONDITIONS 
    # ======================================================== 
 
    future_temperature = forecast[ 
        "Target_Temperature_Plus5h" 
    ] 
 
    future_ph = forecast[ 
        "Target_pH_Plus5h" 
    ] 
 
    future_do = forecast[ 
        "Target_DO_Plus5h" 
    ] 
 
    future_sugar = forecast[ 
        "Target_Sugar_Plus5h" 
    ] 
 
    future_biomass = forecast[ 
        "Target_Biomass_Plus5h" 
    ] 
 
    future_pha = forecast[ 
        "Target_PHA_Accumulation_Plus5h" 
    ] 
 
    future_yield = forecast[ 
        "Target_PHA_Yield_Plus5h" 
    ] 
 
 
    # ======================================================== 
    # ALERT SYSTEM 
    # ======================================================== 
 
    alerts = [] 
 
 
    if future_temperature < 28 or future_temperature > 33: 
 
        alerts.append( 
            "Temperature may move outside the preferred range." 
        ) 
 
 
    if future_ph < 6.5 or future_ph > 7.3: 
 
        alerts.append( 
            "pH may move outside the preferred range." 
        ) 
 
 
    if future_do < 20: 
 
        alerts.append( 
            "Dissolved oxygen may become too low." 
        ) 
 
 
    if future_sugar < 3: 
 
        alerts.append( 
            "Sugar concentration may become limiting." 
        ) 
 
 
    # ======================================================== 
    # STATUS 
    # ======================================================== 
 
    if len(alerts) == 0: 
 
        status = "STABLE" 
 
    elif len(alerts) <= 2: 
 
        status = "WARNING" 
 
    else: 
 
        status = "CRITICAL" 
 
 
    # ======================================================== 
    # FINAL RESPONSE 
    # ======================================================== 
 
    return { 
 
        "Input_Data": { 
 
            "Rice_Straw_Mass_kg": 
                Rice_Straw_Mass, 
 
            "Moisture_percent": 
                Moisture, 
 
            "Initial_Sugar_gL": 
                Initial_Sugar, 
 
            "Temperature_C": 
                Temperature, 
 
            "pH": 
                pH, 
 
            "DO_percent": 
                DO, 
 
            "RPM": 
                RPM, 
 
            "Fermentation_Time_hours": 
                Fermentation_Time 
        }, 
 
 
        # ==================================================== 
        # MODEL 1 
        # ==================================================== 
 
        "Model_1": { 
 
            "Name": 
                "PHA Yield Prediction", 
 
            "Predicted_PHA_Yield_gL": 
                round(predicted_yield, 3) 
        }, 
 
 
        # ==================================================== 
        # MODEL 2 
        # ==================================================== 
 
        "Digital_Twin": { 
 
            "Input_From_Model_1": { 
 
                "Predicted_PHA_Yield_gL": 
                    round(predicted_yield, 3) 
            }, 
 
 
            "Current_Fermentation_State": { 
 
                "Sugar_Concentration_gL": 
                    round(float(sugar_concentration), 2), 
 
                "Biomass_gL": 
                    round(float(biomass), 3), 
 
                "PHA_Accumulation_percent": 
                    round(float(pha_accumulation), 2) 
            }, 
 
 
            "Forecast_After_5_Hours": { 
 
                "Temperature_C": 
                    round(float(future_temperature), 2), 
 
                "pH": 
                    round(float(future_ph), 2), 
 
                "DO_percent": 
                    round(float(future_do), 2), 
 
                "Sugar_Concentration_gL": 
                    round(float(future_sugar), 2), 
 
                "Biomass_gL": 
                    round(float(future_biomass), 3), 
 
                "PHA_Accumulation_percent": 
                    round(float(future_pha), 2), 
 
                "PHA_Yield_gL": 
                    round(float(future_yield), 3) 
            }, 
 
 
            "Status": 
                status, 
 
 
            "Alerts": 
                alerts 
        } 
    } 
 
 
# ============================================================ 
# PREDICT - JSON 
# ============================================================ 
 
@app.post("/predict") 
def predict(data: PHAInput): 
 
    return run_prediction( 
 
        Rice_Straw_Mass=data.Rice_Straw_Mass, 
 
        Moisture=data.Moisture, 
 
        Initial_Sugar=data.Initial_Sugar, 
 
        Temperature=data.Temperature, 
 
        pH=data.pH, 
 
        DO=data.DO, 
 
        RPM=data.RPM, 
 
        Fermentation_Time=data.Fermentation_Time 
    ) 
 
 
# ============================================================ 
# PREDICT FORM 
# ============================================================ 
 
@app.post("/predict-form") 
def predict_form( 
 
    Rice_Straw_Mass: float = Form( 
        ..., 
        gt=0, 
        description="Enter the rice straw mass in kilograms (kg) — Example: 1.5" 
    ), 
 
    Moisture: float = Form( 
        ..., 
        ge=0, 
        le=100, 
        description="Enter the moisture content of the rice straw (%) — Example: 10.5" 
    ), 
 
    Initial_Sugar: float = Form( 
        ..., 
        gt=0, 
        description="Enter the initial sugar concentration (g/L) — Example: 25" 
    ), 
 
    Temperature: float = Form( 
        ..., 
        description="Enter the fermentation temperature (°C) — Example: 30" 
    ), 
 
    pH: float = Form( 
        ..., 
        description="Enter the pH value — Example: 6.8" 
    ), 
 
    DO: float = Form( 
        ..., 
        ge=0, 
        le=100, 
        description="Enter the dissolved oxygen (DO) percentage (%) — Example: 35" 
  ), 
 
    RPM: float = Form( 
        ..., 
        ge=0, 
        description="Enter the mixing speed (RPM) — Example: 180" 
    ), 
 
    Fermentation_Time: float = Form( 
        ..., 
        ge=0, 
        description="Enter the fermentation time (hours) — Example: 20" 
    ) 
 
): 
 
    return run_prediction( 
 
        Rice_Straw_Mass=Rice_Straw_Mass, 
 
        Moisture=Moisture, 
 
        Initial_Sugar=Initial_Sugar, 
 
        Temperature=Temperature, 
 
        pH=pH, 
 
        DO=DO, 
 
        RPM=RPM, 
 
        Fermentation_Time=Fermentation_Time 
    )