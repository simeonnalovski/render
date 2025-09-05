import functools
import logging
import os
import smtplib
from email.mime.text import MIMEText
from io import BytesIO
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
from dotenv import load_dotenv
from evidently.ui.workspace import CloudWorkspace
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Depends
from schema import validate_dataframe
from evidently import Report
from evidently.presets import DataSummaryPreset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

load_dotenv()
# --- Custom Exceptions (For robust error handling) ---
class NameNotFoundError(ValueError):
    def __init__(self, identifier, message="Name not found or is N/A"):
        self.identifier = identifier
        self.message = f"{message} for identifier: '{identifier}'"
        super().__init__(self.message)


class EmptyDataFrameError(ValueError):
    def __init__(self, message="The data source is empty or could not be loaded."):
        self.message = message
        super().__init__(self.message)


# Initialize the FastAPI app
app = FastAPI(
    title="Revenue Forecast API",
    description="API for forecasting revenue using fine-tuned AutoGluon models.",
    version="2.0",
    docs_url="/documentation"
)
BASE_DIR = Path(__file__).resolve().parent
# --- CONFIGURATION ---
MODEL_PATH_TECH = BASE_DIR / "chronos_tiny_Technology_final_forecast_covariates"
MODEL_PATH_COMM = BASE_DIR / "chronos_tiny_Communication Services_final_forecast_covariates"
MODEL_PATH_HEALTH = BASE_DIR / "chronos_tiny_Healthcare_final_forecast_covariates"
MODEL_PATH_ENERGY= BASE_DIR / "chronos_tiny_Energy_final_forecast_covariates"
MODEL_PATH_CONS_DEF= BASE_DIR / "chronos_tiny_Consumer_Defensive_final_forecast_covariates"
# Email configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587



# --- Dependency Function for Token Authentication ---
def verify_token(token: str | None = Query(None, description="Access token for authentication")):
    if token is None or token != os.getenv('SECRET_TOKEN'):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized access: Invalid or missing token."
        )
    return token


# --- World Bank Data Fetching Helpers (Extended) ---
BASE_URL = "http://api.worldbank.org/v2"

# General macroeconomic indicators
GENERAL_INDICATORS = {
    "NY.GDP.MKTP.CD": "GDP (current US$)",
    "NY.GDP.PCAP.CD": "GDP per capita (current US$)",
    "NY.GDP.MKTP.KD.ZG": "GDP growth (annual %)",
    "NE.CON.PRVT.ZS": "Household final consumption (% of GDP)",
}

# Indicators specific to the technology sector
TECHNOLOGY_INDICATORS = {
    "BM.GSR.CMCP.ZS": "Communications, computer, etc. (% of service imports, BoP)",
    "BM.GSR.ROYL.CD": "Charges for the use of intellectual property, payments (BoP, current US$)",
    "BX.GSR.CCIS.CD": "ICT service exports (BoP, current US$)",
    "BX.GSR.CCIS.ZS": "ICT service exports (% of service exports, BoP)",
    "BX.GSR.CMCP.ZS": "Communications, computer, etc. (% of service exports, BoP)"
}

# Indicators specific to the communication services sector
COMMUNICATION_INDICATORS = {
    "IT.CEL.SETS": "Mobile cellular subscriptions",
    "IT.CEL.SETS.P2": "Mobile cellular subscriptions (per 100 people)",
    "IT.NET.USER.ZS": "Individuals using the Internet (% of population)",
    "BX.GSR.ROYL.CD": "Charges for the use of intellectual property, receipts (BoP, current US$)",
    "BM.GSR.ROYL.CD": "Charges for the use of intellectual property, payments (BoP, current US$)"
}

HEALTH_INDICATORS = {
    "SH.XPD.CHEX.GD.ZS": "Health expenditure (% of GDP)",
    "GB.XPD.RSDV.GD.ZS": "Research and development expenditure (% of GDP)",
    "BX.GSR.ROYL.CD": "Charges for the use of intellectual property, receipts (current US$)",
    "IP.PAT.RESD": "Patent applications by residents",
}
ENERGY_INDICATORS = {
    "EG.ELC.PROD.KH": "Electric power consumption (kWh per capita)",
    "EG.IMP.CONS.ZS": "Energy imports, net (% of energy use)",
    "BX.GRM.ENRG.ZS": "Fuel exports (% of merchandise exports)",
    "EG.FEC.RNEW.ZS": "Renewable energy consumption (% of total final energy consumption)",
    "NY.GDP.DEFL.ZS.AD": "GDP deflator (annual %)",
    "SP.RRS.OIL.MA": "Crude oil price (WTI, annual)",
}
CONSUMER_DEFENSIVE_INDICATORS = {
    "FP.CPI.TOTL.ZG": "Inflation, consumer prices (annual %)",
    "NE.CON.PRVT.PC.KD": "Household final consumption expenditure per capita (constant 2017 US$)",
    "SH.XPD.CHEX.GD.ZS": "Current health expenditure (% of GDP)",
    "AG.PRD.FOOD.XD": "Food production index (2014-2016 = 100)",
}
@functools.lru_cache(maxsize=1)
def get_world_bank_official_map():
    url = f"{BASE_URL}/country?format=json&per_page=500"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    if not data or len(data) < 2 or not data[1]:
        raise ValueError("Invalid World Bank country map data.")
    geo_map = {}
    for entry in data[1]:
        country_id = entry.get('id')
        country_name = entry.get('name')
        region_info = entry.get('region')
        is_country = (region_info and region_info.get('id') != 'NAX' and
                      entry.get('incomeLevel', {}).get('id') != 'NAX' and
                      entry.get('lendingType', {}).get('id') != 'NAX')
        if country_name:
            geo_map[country_name.strip().title()] = {
                'id': country_id,
                'region_id': region_info.get('id') if region_info else None,
                'is_country': is_country
            }
    return geo_map


def get_wb_codes_from_country_name(country_name):
    official_wb_map = get_world_bank_official_map()
    normalized_name = country_name.strip().title()
    entity_info = official_wb_map.get(normalized_name)
    if entity_info and entity_info['is_country']:
        return entity_info['id'], entity_info['region_id']
    raise ValueError(f"Unmapped World Bank country: {normalized_name}")


def fetch_indicator_data(entity_code, indicator_code, label, is_region=False):
    column_suffix = "_region" if is_region else "_country"
    url = f"{BASE_URL}/country/{entity_code}/indicator/{indicator_code}?format=json&per_page=1000"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if not data or len(data) < 2 or not data[1]:
            return pd.DataFrame()
        records = [rec for rec in data[1] if rec["value"] is not None]
        if not records:
            return pd.DataFrame()
        return pd.DataFrame([{
            "year": int(rec["date"]),
            f"{label}{column_suffix}": rec["value"]
        } for rec in records])
    except requests.exceptions.RequestException:
        return pd.DataFrame()


def fetch_all_covariates_from_params(country: str, indicators: dict):
    try:
        country_code, region_code = get_wb_codes_from_country_name(country)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    all_indicators = {**GENERAL_INDICATORS, **indicators}
    df_all = pd.DataFrame({"year": []})

    for code, label in all_indicators.items():
        try:
            c_data = fetch_indicator_data(country_code, code, label, is_region=False)
            if not c_data.empty:
                df_all = pd.merge(df_all, c_data, on="year", how="outer") if not df_all.empty else c_data
        except Exception:
            pass
        try:
            r_data = fetch_indicator_data(region_code, code, label, is_region=True)
            if not r_data.empty:
                df_all = pd.merge(df_all, r_data, on="year", how="outer")
        except Exception:
            pass

    if df_all.empty:
        raise HTTPException(status_code=500, detail="Could not retrieve any World Bank data for the specified country.")

    df_all = df_all.sort_values(by="year", ascending=True)
    for col in df_all.columns:
        if col != 'year':
            df_all[col] = df_all[col].interpolate(method='linear', limit_direction='both')
            df_all[col] = df_all[col].fillna(df_all[col].mean())

    return df_all


@app.on_event("startup")
def load_resources():
    global predictor_tech, predictor_comm, predictor_health,predictor_energy, predictor_con_def
    try:
        predictor_tech = TimeSeriesPredictor.load(MODEL_PATH_TECH)
        logger.info("Technology model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading the Technology model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error loading the Technology model: {e}")
    try:
        predictor_comm = TimeSeriesPredictor.load(MODEL_PATH_COMM)
        logger.info("Communication Services model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading the Communication Services model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error loading the Communication Services model: {e}")

    try:
        predictor_health= TimeSeriesPredictor.load(MODEL_PATH_HEALTH)
        logger.info("Healthcare model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading the Healthcare model: {e}",exc_info=True)

    try:
        predictor_energy=TimeSeriesPredictor.load(MODEL_PATH_ENERGY)
        logger.info("Energy model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading Energy model: {e}",exc_info=True)

    try:
         predictor_con_def=TimeSeriesPredictor.load(MODEL_PATH_CONS_DEF)
         logger.info("Consumer Defensive model loaded successfully")
    except Exception as e:
         logger.error(f"Error loading Consumer Defensive model: {e}", exc_info=True)


def send_alert_email(subject, body):
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = os.getenv('EMAIL_SENDER')
        msg["To"] = os.getenv('EMAIL_RECEIVER')

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(os.getenv('EMAIL_SENDER'), os.getenv('EMAIL_PASSWORD'))
            server.sendmail(os.getenv('EMAIL_SENDER'), os.getenv('EMAIL_RECEIVER'), msg.as_string())
        logger.info("Uncertainty alert email sent successfully!")
    except Exception as e:
        logger.error(f"Failed to send email alert: {e}")


def process_prediction(file_contents: bytes, country: str, indicators: dict, predictor: TimeSeriesPredictor,
                       forecast_years: int, sector: str):
    try:
        df_user = pd.read_csv(BytesIO(file_contents))

        if not validate_dataframe(df_user):
            raise HTTPException( status_code= 422, detail="The given CSV doeen't match the required inocme statement schema")
        report = Report([DataSummaryPreset()], tags=[sector])
        my_eval = report.run(df_user)
        ws = CloudWorkspace(
            token=os.getenv("EVIDENTLY_API_KEY"),
            url="https://app.evidently.cloud"
        )
        project = ws.get_project(os.getenv("EVIDENTLY_PROJECT_ID"))
        ws.add_run(project.id, my_eval, include_data=False)


        try:
            df_user['date'] = pd.to_datetime(df_user['date'])
            df_user['revenue'] = pd.to_numeric(df_user['revenue'])
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail="The 'date' or 'revenue' column contains invalid data formats.")


        df_user['date'] = pd.to_datetime(df_user['date'])
        df_user['year'] = df_user['date'].dt.year
        df_user['item_id'] = 'user_data'
        df_user.rename(columns={'revenue': 'target', 'date': 'timestamp'}, inplace=True)
        df_user = df_user.sort_values(by="timestamp").reset_index(drop=True)

        df_covariates = fetch_all_covariates_from_params(country, indicators)
        df_full = pd.merge(df_user, df_covariates, on='year', how='left')

        df_full['timestamp'] = pd.to_datetime(df_full['year'], format='%Y')
        if df_full.shape[0] <5:
            raise HTTPException(status_code=400, detail="Insuficient data for internal evaluation")
        elif df_full.shape[0] == 5:
            eval_data = df_full
            historical_data_for_prediction = df_full
            historical_ts = TimeSeriesDataFrame(historical_data_for_prediction.set_index(['item_id', 'timestamp']))
            known_covariates = TimeSeriesDataFrame(eval_data.set_index(['item_id', 'timestamp']))
        else:
            eval_data = df_full.iloc[-5:]
            historical_data_for_prediction=df_full.iloc[-5:]
            historical_ts = TimeSeriesDataFrame(historical_data_for_prediction.set_index(['item_id', 'timestamp']))
            known_covariates = TimeSeriesDataFrame(eval_data.set_index(['item_id', 'timestamp']))

        # Predict on the holdout period
        eval_predictions = predictor.predict(
            historical_ts,
            known_covariates=known_covariates
        )
        actual_values = eval_data['target'].values
        forecast_values = eval_predictions['mean'].values
        mape = np.mean(np.abs((actual_values - forecast_values) / actual_values)) * 100

        covariate_cols = [c for c in df_full.columns if c not in ['year', 'target', 'item_id', 'timestamp']]
        for col in covariate_cols:
            df_full[col] = df_full[col].ffill().bfill()

        last_year = df_full['year'].max()
        future_years = range(last_year + 1, last_year + 1 + forecast_years)
        df_future_covariates = pd.DataFrame({'year': future_years})
        df_future_covariates['timestamp'] = pd.to_datetime(df_future_covariates['year'], format='%Y')
        df_future_covariates['item_id'] = 'user_data'

        logger.info("User data validated and pre-processed.")

        for col in covariate_cols:
            df_future_covariates[col] = df_full[col].iloc[-1]

        historical_data = TimeSeriesDataFrame(
            df_full[['item_id', 'timestamp', 'target'] + covariate_cols].set_index(['item_id', 'timestamp'])
        )

        future_covariates = TimeSeriesDataFrame(
            df_future_covariates[['item_id', 'timestamp'] + covariate_cols].set_index(['item_id', 'timestamp'])
        )

        predictions = predictor.predict(historical_data, known_covariates=future_covariates)
        # --- The fix: Trim the predictions to the requested length
        predictions_trimmed = predictions.loc['user_data'].iloc[:forecast_years]
        predictions_values = predictions_trimmed.values

        # Use the trimmed predictions for all subsequent calculations
        # This is the corrected section to handle the `float` object error
        quantiles = np.quantile(predictions_values, [0.1, 0.5, 0.9], axis=1)

        if quantiles.ndim == 1:
            lower_bound_80, median_forecast_np, upper_bound_80 = quantiles
            lower_bound_80 = np.array([lower_bound_80])
            median_forecast_np = np.array([median_forecast_np])
            upper_bound_80 = np.array([upper_bound_80])
        else:
            lower_bound_80, median_forecast_np, upper_bound_80 = quantiles[0, :], quantiles[1, :], quantiles[2, :]

        forecast_years_list = [y.year for y in predictions_trimmed.index]

        # Ensure the lists are not empty to avoid division by zero
        if not median_forecast_np.size or not upper_bound_80.size or not lower_bound_80.size:
            raise HTTPException(status_code=500, detail="Prediction resulted in empty data.")

        # Calculate key metrics based on the trimmed data
        relative_uncertainty_80 = ((upper_bound_80 - lower_bound_80) / median_forecast_np * 100).tolist()

        if relative_uncertainty_80:
            avg_uncertainty_80 = sum(relative_uncertainty_80) / len(relative_uncertainty_80)
        else:
            avg_uncertainty_80 = 0.0

        if avg_uncertainty_80 > float(os.getenv('THRESHOLD')):
            subject = f"Revenue Forecast Alert: High Uncertainty ({avg_uncertainty_80:.2f}%)"
            body = (f"The average uncertainty for the latest revenue forecast exceeds {float(os.getenv('THRESHOLD')):.1f}%\n\n"
                    f"Average Uncertainty (80% CI): {avg_uncertainty_80:.2f}%")
            send_alert_email(subject, body)

        logger.info("Prediction successful.")
        return {
            "valiadtion_MAPE":f"{mape:.2f}%" if mape is not None else "N/A",
            "forecast_start_year": forecast_years_list[0],
            "prediction_length": forecast_years,
            "forecasted_revenue": dict(zip(forecast_years_list, median_forecast_np.tolist())),
            "confidence_metrics": {
                "avg_uncertainty_80_percent": f"{avg_uncertainty_80:.2f}%",
                "relative_uncertainty_by_year_percent": {
                    "80_percent_ci": dict(zip(forecast_years_list, [f"{v:.2f}%" for v in relative_uncertainty_80]))
                }
            }
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error during prediction: {e}")


@app.post("/predict")
async def predict_revenue(
        file: UploadFile = File(...),
        country: str = Query(..., description="Country for macroeconomic data"),
        sector: str = Query(..., description="Sector ('Technology' or 'Communication Services')"),
        forecast_years: int = Query(..., description="Number of years to forecast", ge=1),
        token: str = Depends(verify_token)
):
    """
    Accepts a CSV file of historical revenue data and returns a future forecast.
    The number of years to forecast is specified by the 'forecast_years' query parameter.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file format. Upload a CSV.")

    if forecast_years > 5:
        raise HTTPException(status_code=400, detail="Due to conserns with prediction quality, the maximum limit for forecast_years is 5")

    contents = await file.read()
    sector_lower = sector.lower().replace(" ", "_")

    if sector_lower == 'technology':
        if not predictor_tech:
            raise HTTPException(status_code=503, detail="Technology predictor not loaded.")
        return process_prediction(contents, country, TECHNOLOGY_INDICATORS, predictor_tech, forecast_years,sector_lower)

    elif sector_lower == 'communication_services':
        if not predictor_comm:
            raise HTTPException(status_code=503, detail="Communication Services predictor not loaded.")
        return process_prediction(contents, country, COMMUNICATION_INDICATORS, predictor_comm, forecast_years,sector_lower)

    elif sector_lower == 'healthcare':
        if not predictor_health:
            raise HTTPException(status_code=503, detail="Healthcare predictor not loaded")
        return  process_prediction(contents,country,HEALTH_INDICATORS,predictor_health,forecast_years,sector_lower)
    elif sector_lower == 'energy':
        if not predictor_energy:
            raise HTTPException(status_code=503,detail="Energy predictor not loaded")
        return process_prediction(contents,country,ENERGY_INDICATORS,predictor_energy,forecast_years,sector_lower)
    elif sector_lower == 'consumer_defensive':
        if not predictor_con_def:
            raise HTTPException(status_code=503,detail="Consumer Defensive model not loaded")
        return process_prediction(contents,country,CONSUMER_DEFENSIVE_INDICATORS,predictor_con_def,forecast_years,sector_lower)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported sector: '{sector}'.")


@app.get("/health")
def health_check():
    logger.info("Health check request received.")
    return {
        "status": "ok",
        "technology_model_loaded": predictor_tech is not None,
        "communication_model_loaded": predictor_comm is not None,
        "healthcare_model_loaded":  predictor_health is not None,
        "energy_model_loaded": predictor_energy is not None,
        "con_def_model_loaded": predictor_con_def is not None
    }