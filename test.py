import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import traceback
import json 
import requests


N_HOURS_DEFAULT = 24 
MODEL_DIR = '.' 
BASELINE_STATS_PATH = os.path.join(MODEL_DIR, 'baseline_stats.csv')
PREDICTION_API_URL = "http://127.0.0.1:5000/predict" 


try:
    baseline_stats = pd.read_csv(BASELINE_STATS_PATH, index_col=0)
    baseline_mean = baseline_stats['mean']
    baseline_std = baseline_stats['std']
    print("Baseline stats loaded successfully for Synthetic Data API.")
except FileNotFoundError:
    print(f"Error: Baseline stats file not found at '{BASELINE_STATS_PATH}'. Using default values.")
    baseline_mean = pd.Series({'temperature': 20, 'humidity': 55, 'CO2CosIRValue': 70, 'CO2MG811Value': 510, 'MOX1': 540, 'MOX2': 720, 'MOX3': 670, 'MOX4': 620, 'COValue': 115})
    baseline_std = pd.Series({'temperature': 1.5, 'humidity': 2.5, 'CO2CosIRValue': 20, 'CO2MG811Value': 10, 'MOX1': 20, 'MOX2': 20, 'MOX3': 20, 'MOX4': 25, 'COValue': 20})
except Exception as e:
    print(f"Error loading baseline stats: {e}. Using default values.")
    baseline_mean = pd.Series({'temperature': 20, 'humidity': 55, 'CO2CosIRValue': 70, 'CO2MG811Value': 510, 'MOX1': 540, 'MOX2': 720, 'MOX3': 670, 'MOX4': 620, 'COValue': 115})
    baseline_std = pd.Series({'temperature': 1.5, 'humidity': 2.5, 'CO2CosIRValue': 20, 'CO2MG811Value': 10, 'MOX1': 20, 'MOX2': 20, 'MOX3': 20, 'MOX4': 25, 'COValue': 20})


EXPECTED_API_COLS = [
    'temperature', 'humidity', 'CO2CosIRValue', 'CO2MG811Value',
    'MOX1', 'MOX2', 'MOX3', 'MOX4', 'COValue',
    'hour', 'day_of_week', 'any_activity'
]

ENV_COLS_FOR_STD = baseline_mean.index.tolist()


def generate_synthetic_data(num_hours):
    """Generates synthetic hourly sensor data with optional random anomalies."""
    print(f"\n--- Generating {num_hours} hours of synthetic data ---")
    timestamps = pd.date_range(start='2024-01-10 00:00:00', periods=num_hours, freq='h')
    df = pd.DataFrame(index=timestamps)

    
    print("  Generating base sensor values...")
    for col in ENV_COLS_FOR_STD:
        mean = baseline_mean.get(col, 0)
        std = baseline_std.get(col, 1)
        df[col] = np.random.normal(loc=mean, scale=std/3 if std > 0 else 0.1, size=num_hours)

    print("  Generating time features...")
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek

    print("  Generating activity simulation...")
    active_prob = np.sin(np.pi * (df['hour'] - 6) / 17)
    active_prob = np.clip(active_prob, 0.1, 0.9)
    df['any_activity'] = (np.random.rand(num_hours) < active_prob).astype(float)

    
    print("  Attempting to introduce anomalies randomly...")
    injected_anomalies_log = []
    low_temp_prob = 0.5; high_co_prob = 0.5; inactivity_prob = 0.4

    
    if np.random.rand() < low_temp_prob:
        duration = 3
        if num_hours >= duration:
            start_idx = np.random.randint(0, num_hours - duration + 1)
            end_idx = start_idx + duration - 1
            df.iloc[start_idx:end_idx+1, df.columns.get_loc('temperature')] = 17.0 + (np.random.rand(duration) * 0.5 - 0.25)
            log_msg = f"Low temp injected at hours {start_idx}-{end_idx}"
            print(f"    - {log_msg}")
            injected_anomalies_log.append(log_msg)

    
    if np.random.rand() < high_co_prob:
        duration = 2
        if num_hours >= duration:
            start_idx = np.random.randint(0, num_hours - duration + 1)
            end_idx = start_idx + duration - 1
            co_z4_value = baseline_mean.get('COValue', 115) + 4 * baseline_std.get('COValue', 20)
            df.iloc[start_idx:end_idx+1, df.columns.get_loc('COValue')] = co_z4_value + np.random.rand(duration)
            log_msg = f"High CO injected at hours {start_idx}-{end_idx}"
            print(f"    - {log_msg}")
            injected_anomalies_log.append(log_msg)

    
    if np.random.rand() < inactivity_prob:
        duration = 5
        min_end_hour = 10 + (duration -1); max_end_hour = min(22, num_hours - 1)
        if max_end_hour >= min_end_hour :
            end_hour = np.random.randint(min_end_hour, max_end_hour + 1)
            end_idx = end_hour; start_idx = end_idx - duration + 1
            if start_idx >= 0:
                 df.iloc[start_idx:end_idx+1, df.columns.get_loc('any_activity')] = 0.0
                 log_msg = f"Inactivity injected for hours {start_idx}-{end_idx} (ends at hour {end_hour})"
                 print(f"    - {log_msg}")
                 injected_anomalies_log.append(log_msg)
            else: print(f"    - Skipped inactivity injection: Calculated start index {start_idx} was invalid.")
        else: print(f"    - Skipped inactivity injection: Not enough hours ({num_hours}) to fit inactivity ending between 10-22.")

    if not injected_anomalies_log: print("  - No anomalies were randomly injected for this run.")

    
    for col in EXPECTED_API_COLS:
        if col not in df.columns: df[col] = 0
    df_payload = df[EXPECTED_API_COLS].copy()
    
    df_payload['timestamp'] = df_payload.index.strftime('%Y-%m-%d %H:%M:%S')

    print("Synthetic data generation complete.")
    
    return df_payload, injected_anomalies_log



app = Flask(__name__)
CORS(app) 

@app.route('/trigger_simulation', methods=['GET'])
def trigger_simulation_and_predict():
    """
    Generates synthetic data, sends it to the prediction API,
    and returns BOTH the generated data and the prediction API's response.
    """
    print("\n=== /trigger_simulation request received ===")
    prediction_results = None 
    injected_anomalies = [] 
    synthetic_data_list = [] 

    try:
        
        num_hours_str = request.args.get('hours', str(N_HOURS_DEFAULT))
        try:
            num_hours = int(num_hours_str)
            if num_hours <= 0 or num_hours > 1000: raise ValueError("Hours must be 1-1000.")
        except ValueError as e:
             print(f"Error: Invalid 'hours' parameter: {num_hours_str}. {e}")
             return jsonify({"error": f"Invalid 'hours' parameter: {num_hours_str}. Must be a positive integer (max 1000)."}), 400

        synthetic_payload_df, injected_anomalies = generate_synthetic_data(num_hours)
        
        payload_list_for_prediction = synthetic_payload_df[EXPECTED_API_COLS].to_dict(orient='records')
        
        synthetic_data_list = synthetic_payload_df.to_dict(orient='records')


        
        print(f"\nForwarding generated data to Prediction API: {PREDICTION_API_URL}")
        try:
            pred_response = requests.post(PREDICTION_API_URL, json=payload_list_for_prediction, timeout=30)
            pred_response.raise_for_status()
            prediction_results = pred_response.json() 
            print(f"Prediction API returned status code: {pred_response.status_code}")

        except requests.exceptions.ConnectionError as e:
             print(f"Error: Could not connect to Prediction API at {PREDICTION_API_URL}. Is it running?")
             
             return jsonify({
                 "generated_data": synthetic_data_list,
                 "prediction_error": f"Could not connect to Prediction API: {e}",
                 "injected_anomalies_log": injected_anomalies
             }), 503
        except requests.exceptions.Timeout:
             print(f"Error: Request to Prediction API timed out.")
             return jsonify({
                 "generated_data": synthetic_data_list,
                 "prediction_error": "Prediction API request timed out",
                 "injected_anomalies_log": injected_anomalies
             }), 504
        except requests.exceptions.RequestException as e:
             print(f"Error calling Prediction API: {e}")
             error_detail = "No details available."
             try:
                 error_detail = pred_response.json().get("error", pred_response.text)
             except: error_detail = pred_response.text if hasattr(pred_response, 'text') else str(pred_response)
             return jsonify({
                  "generated_data": synthetic_data_list,
                  "prediction_error": f"Prediction API call failed: {e}",
                  "prediction_details": error_detail,
                  "injected_anomalies_log": injected_anomalies
             }), 502

        
        print("Combining generated data and prediction results for response.")
        final_response = {
            "generated_data": synthetic_data_list,
            "prediction_results": prediction_results,
            "injected_anomalies_log": injected_anomalies
        }
        return jsonify(final_response)

    except Exception as e:
        print(f"!!! Error during simulation trigger: {e}")
        traceback.print_exc()
        
        return jsonify({
            "error": "An internal error occurred during simulation trigger.",
            "details": str(e),
            "generated_data": synthetic_data_list, 
            "injected_anomalies_log": injected_anomalies
            }), 500

if __name__ == '__main__':
    
    
    app.run(debug=True, port=5001, use_reloader=False)

