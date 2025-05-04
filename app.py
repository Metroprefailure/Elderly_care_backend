import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
import os
import traceback 
import smtplib
from email.mime.text import MIMEText
import datetime 

from flask_cors import CORS


app = Flask(__name__)
CORS(app) 

# Email Configuration
EMAIL_SENDER = 'alerttestingelderlymonitoring@gmail.com'
EMAIL_PASSWORD = 'qlwu kapw kqfv ntnm' # Use App Password if 2FA is enabled
EMAIL_RECIPIENT = 'jsvighnesh@gmail.com'
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587 


MODEL_DIR = '.' 
IFOREST_MODEL_PATH = os.path.join(MODEL_DIR, 'isolation_forest_model.joblib')
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, 'lstm_autoencoder_model.keras')
SCALER_PATH = os.path.join(MODEL_DIR, 'min_max_scaler.joblib')
BASELINE_STATS_PATH = os.path.join(MODEL_DIR, 'baseline_stats.csv')
TIMESTEPS = 12 

LSTM_MAE_THRESHOLD = 0.2099 


print("Loading models and objects...")
try:
    iso_forest = joblib.load(IFOREST_MODEL_PATH)
    lstm_model = load_model(LSTM_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    baseline_stats = pd.read_csv(BASELINE_STATS_PATH, index_col=0)
    baseline_mean = baseline_stats['mean']
    baseline_std = baseline_stats['std']
    print("Models and objects loaded successfully.")
    
    try:
        SCALER_FEATURE_COLS = scaler.feature_names_in_.tolist()
        print(f"Scaler expects features (from scaler.feature_names_in_): {SCALER_FEATURE_COLS}")
        print(f"Number of features expected by scaler: {scaler.n_features_in_}")
    except AttributeError:
        SCALER_FEATURE_COLS = [
            'standardized_temperature', 'standardized_humidity',
            'standardized_CO2CosIRValue', 'standardized_CO2MG811Value',
            'standardized_MOX1', 'standardized_MOX2', 'standardized_MOX3', 'standardized_MOX4',
            'standardized_COValue',
            'hour', 'day_of_week', 'any_activity'
        ]
        print(f"Warning: Scaler has no 'feature_names_in_'. Assuming expected features are: {SCALER_FEATURE_COLS}")
    
except FileNotFoundError as e:
    print(f"Error loading file: {e}. Make sure all model/object files are in the '{MODEL_DIR}' directory.")
    exit()
except Exception as e:
    print(f"Error loading models/objects: {e}")
    traceback.print_exc()
    exit()


ORIGINAL_FEATURE_COLS = [
    'temperature', 'humidity', 'CO2CosIRValue', 'CO2MG811Value',
    'MOX1', 'MOX2', 'MOX3', 'MOX4', 'COValue',
    'hour', 'day_of_week', 'any_activity'
]

ENV_COLS_FOR_STD = baseline_mean.index.tolist()






def create_sequences(data, timesteps):
    """Creates sequences for LSTM input."""
    X = []
    if isinstance(data, pd.DataFrame): data = data.values
    elif isinstance(data, pd.Series): data = data.values.reshape(-1, 1)
    if data.ndim == 1: data = data.reshape(-1, 1)

    n_features = data.shape[1]
    print(f"  Creating sequences from data shape: {data.shape}") 
    for i in range(len(data) - timesteps + 1):
        X.append(data[i:(i + timesteps)])
    result = np.array(X)
    print(f"  Resulting sequence shape: {result.shape}") 
    
    if result.ndim != 3 or (result.size > 0 and (result.shape[1] != timesteps or result.shape[2] != n_features)):
         print(f"  Warning: Unexpected sequence shape created: {result.shape}. Expected: (?, {timesteps}, {n_features})")
    return result

def apply_rule_based_detection(df_input_with_std):
    """
    Applies the refined rule-based logic.
    Assumes Z-score columns (standardized_*) are already present.
    Input: DataFrame with original sensor values AND Z-score standardized columns.
    Output: DataFrame with added 'anomaly_*' columns for rules.
    """
    df = df_input_with_std.copy() 
    print("  Rule Func: Applying rule-based logic (using existing Z-scores)...") 

    
    
    co_thresh = 3.5; mox_thresh = 4.0; gas_duration_window = 2
    df['anomaly_high_gas'] = False
    if 'standardized_COValue' in df.columns:
        if df['standardized_COValue'].notna().sum() >= gas_duration_window:
            cond_co = (df['standardized_COValue'] > co_thresh).rolling(window=gas_duration_window, min_periods=gas_duration_window).sum() == gas_duration_window
            df['anomaly_high_gas'] = df['anomaly_high_gas'] | cond_co.fillna(False)
    mox_cols_std = [f'standardized_MOX{i}' for i in range(1, 5) if f'standardized_MOX{i}' in df.columns]
    if mox_cols_std:
        if df[mox_cols_std].notna().sum().max() >= gas_duration_window:
             cond_mox = (df[mox_cols_std] > mox_thresh).any(axis=1).rolling(window=gas_duration_window, min_periods=gas_duration_window).sum() == gas_duration_window
             df['anomaly_high_gas'] = df['anomaly_high_gas'] | cond_mox.fillna(False)

    
    co2_thresh = 4.0; vent_duration_window = 3
    df['anomaly_poor_vent'] = False
    co2_cols_std = [col for col in ['standardized_CO2CosIRValue', 'standardized_CO2MG811Value'] if col in df.columns]
    if co2_cols_std:
         if df[co2_cols_std].notna().sum().max() >= vent_duration_window:
            cond_co2 = (df[co2_cols_std] > co2_thresh).any(axis=1).rolling(window=vent_duration_window, min_periods=vent_duration_window).sum() == vent_duration_window
            df['anomaly_poor_vent'] = cond_co2.fillna(False)

    
    temp_low_abs = 18.0; temp_low_std_thresh = -3.0; temp_duration_window = 3
    df['anomaly_low_temp'] = False
    if 'temperature' in df.columns and 'standardized_temperature' in df.columns:
        if df[['temperature', 'standardized_temperature']].notna().all(axis=1).sum() >= temp_duration_window:
            cond_temp_low = (((df['temperature'] < temp_low_abs) | (df['standardized_temperature'] < temp_low_std_thresh))
                             .rolling(window=temp_duration_window, min_periods=temp_duration_window).sum() == temp_duration_window)
            df['anomaly_low_temp'] = cond_temp_low.fillna(False)

    
    temp_high_abs = 28.0 
    df['anomaly_high_temp'] = False
    if 'temperature' in df.columns:
         if df['temperature'].notna().sum() >= temp_duration_window:
            cond_temp_high = (df['temperature'] > temp_high_abs).rolling(window=temp_duration_window, min_periods=temp_duration_window).sum() == temp_duration_window
            df['anomaly_high_temp'] = cond_temp_high.fillna(False)

    
    inactivity_duration_window = 5; start_hour = 10; end_hour = 22
    df['anomaly_inactivity'] = False
    if 'any_activity' in df.columns and 'hour' in df.columns:
         if df['any_activity'].notna().sum() >= inactivity_duration_window:
            is_inactive = df['any_activity'] == 0
            cond_inactivity_rolling = (is_inactive.rolling(window=inactivity_duration_window, min_periods=inactivity_duration_window).sum() == inactivity_duration_window)
            is_new_daytime_end = (df['hour'] >= start_hour) & (df['hour'] <= end_hour)
            cond_inactivity = cond_inactivity_rolling & is_new_daytime_end
            df['anomaly_inactivity'] = cond_inactivity.fillna(False)

    
    rule_cols = [col for col in df.columns if col.startswith('anomaly_')]
    print(f"  Rule Func: Returning columns: {rule_cols}") 
    return df[rule_cols]


def calculate_alert_levels(df_with_flags):
    """
    Calculates Alert Levels 1 and 2 based on anomaly flags.
    Input: DataFrame containing ALL individual anomaly flags.
    Output: DataFrame with added 'alert_level_1' and 'alert_level_2' columns.
    """
    df = df_with_flags.copy()
    print("  Alert Func: Calculating alert levels...") 
    critical_rule_cols = ['anomaly_low_temp', 'anomaly_high_gas', 'anomaly_inactivity']
    existing_critical_cols = [col for col in critical_rule_cols if col in df.columns]

    if not existing_critical_cols:
        df['critical_rule_triggered'] = False
        print("  Alert Func: No critical rule columns found.") 
    else:
        for col in existing_critical_cols:
            if col in df.columns: df[col] = df[col].astype(bool)
        df['critical_rule_triggered'] = df[existing_critical_cols].any(axis=1)
        print(f"  Alert Func: Critical rules triggered sum: {df['critical_rule_triggered'].sum()}") 

    all_anomaly_flags = [col for col in df.columns if col.startswith('anomaly_')]
    existing_anomaly_flags = [col for col in all_anomaly_flags if col in df.columns]
    print(f"  Alert Func: Existing anomaly flags for sum check: {existing_anomaly_flags}") 

    for col in existing_anomaly_flags:
         if col in df.columns: df[col] = df[col].astype(bool)

    
    df['alert_level_1'] = False
    if 'anomaly_poor_vent' in df.columns: df['alert_level_1'] = df['alert_level_1'] | df['anomaly_poor_vent']
    if 'anomaly_iforest' in df.columns: df['alert_level_1'] = df['alert_level_1'] | (df['anomaly_iforest'] & (~df['critical_rule_triggered']))
    if 'anomaly_lstm' in df.columns: df['alert_level_1'] = df['alert_level_1'] | (df['anomaly_lstm'] & (~df['critical_rule_triggered']))
    print(f"  Alert Func: Level 1 alerts sum: {df['alert_level_1'].sum()}") 

    
    df['alert_level_2'] = False
    df['alert_level_2'] = df['alert_level_2'] | df['critical_rule_triggered']
    if existing_anomaly_flags:
        anomaly_sum = df[existing_anomaly_flags].sum(axis=1)
        print(f"  Alert Func: Max anomaly sum per row: {anomaly_sum.max()}") 
        df['alert_level_2'] = df['alert_level_2'] | (anomaly_sum >= 2)
    print(f"  Alert Func: Level 2 alerts sum: {df['alert_level_2'].sum()}") 

    return df[['alert_level_1', 'alert_level_2']]

def send_alert_email(subject, body):
    """Sends an email using the configured settings."""
    message = MIMEText(body)
    message['Subject'] = subject
    message['From'] = EMAIL_SENDER
    message['To'] = EMAIL_RECIPIENT

    try:
        print(f"  Email Func: Attempting to send email to {EMAIL_RECIPIENT}...")
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls() # Secure the connection
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECIPIENT, message.as_string())
        success_msg = f"Email sent successfully to {EMAIL_RECIPIENT}."
        print(f"  Email Func: {success_msg}")
        return True, success_msg
    except smtplib.SMTPAuthenticationError as e:
        error_msg = f"Failed to send email. Authentication error for {EMAIL_SENDER}. Check credentials/app password."
        print(f"  Email Func: {error_msg}")
        return False, error_msg
    except smtplib.SMTPConnectError as e:
        error_msg = f"Failed to send email. Could not connect to SMTP server {SMTP_SERVER}:{SMTP_PORT}."
        print(f"  Email Func: {error_msg}")
        return False, error_msg
    except Exception as e:
        error_msg = f"Failed to send email to {EMAIL_RECIPIENT}. Error: {e}"
        print(f"  Email Func: {error_msg}")
        traceback.print_exc()
        return False, error_msg



app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_anomalies():
    """API endpoint to predict anomalies from sensor data."""
    print("\n=== New Request Received ===")
    try:
        input_data = request.get_json()
        if not input_data:
             print("Error: No input data received")
             return jsonify({"error": "No input data received"}), 400
        print(f"Data received type: {type(input_data)}")
        if isinstance(input_data, list) and input_data:
             print(f"First data item: {input_data[0]}")

        
        original_index = None
        if isinstance(input_data, list) and input_data and isinstance(input_data[0], dict) and 'timestamp' in input_data[0]:
             try:
                  original_index = pd.to_datetime([item.get('timestamp') for item in input_data])
                  print("Original timestamps parsed.")
             except Exception as e:
                  print(f"Warning: Could not parse timestamps from input: {e}")

        df = pd.DataFrame(input_data)
        if original_index is not None and len(original_index) == len(df):
             df.index = original_index 

        print(f"Input data converted to DataFrame shape: {df.shape}")
        if df.empty:
             print("Error: Empty DataFrame created from input data")
             return jsonify({"error": "Empty DataFrame created from input data"}), 400

        
        essential_cols = ['temperature', 'humidity', 'CO2CosIRValue', 'CO2MG811Value', 'COValue', 'any_activity']
        missing_essentials = [col for col in essential_cols if col not in df.columns]
        if missing_essentials:
             print(f"Error: Missing essential columns: {missing_essentials}")
             return jsonify({"error": f"Missing essential columns in input data: {missing_essentials}"}), 400

        if 'hour' not in df.columns:
             if isinstance(df.index, pd.DatetimeIndex):
                  print("Generating 'hour' feature from DatetimeIndex...")
                  df['hour'] = df.index.hour
             else:
                  print("Generating 'hour' feature using row index...")
                  df['hour'] = df.index % 24
        if 'day_of_week' not in df.columns:
             if isinstance(df.index, pd.DatetimeIndex):
                  print("Generating 'day_of_week' feature from DatetimeIndex...")
                  df['day_of_week'] = df.index.dayofweek
             else:
                  print("Generating 'day_of_week' feature using row index...")
                  df['day_of_week'] = (df.index // 24) % 7

        
        print("Calculating Z-scores...")
        standardized_cols_generated = []
        df_processed = df.copy() 
        for col in ENV_COLS_FOR_STD:
            if col in df_processed.columns:
                mean = baseline_mean.get(col)
                std = baseline_std.get(col)
                if mean is not None and std is not None and std != 0:
                    std_col_name = f'standardized_{col}'
                    df_processed[std_col_name] = (df_processed[col] - mean) / std
                    standardized_cols_generated.append(std_col_name)
                else:
                    print(f"    - Warning: Baseline stats missing or std=0 for {col}. Cannot calculate Z-score.")
                    df_processed[f'standardized_{col}'] = 0 
            else:
                 print(f"    - Warning: Column {col} not found for Z-score calculation.")
        print(f"Z-score columns generated: {standardized_cols_generated}")

        
        print("Running rule-based detection...")
        
        rule_flags_df = apply_rule_based_detection(df_processed)
        print(f"Rule-based detection finished. Flags shape: {rule_flags_df.shape}")

        
        print("Preparing data for ML models (using scaler's expected columns)...")
        missing_scaler_features = [col for col in SCALER_FEATURE_COLS if col not in df_processed.columns]
        if missing_scaler_features:
             print(f"Fatal Error: Missing columns required by scaler: {missing_scaler_features}")
             return jsonify({"error": f"Missing columns required by scaler: {missing_scaler_features}"}), 400

        df_ordered_for_scaling = df_processed[SCALER_FEATURE_COLS]
        print(f"Shape of data selected for scaling: {df_ordered_for_scaling.shape}")

        if df_ordered_for_scaling.isnull().values.any():
             print("NaN values found in ML features before scaling. Filling with 0...")
             df_ordered_for_scaling = df_ordered_for_scaling.fillna(0)
        else:
             print("No NaN values found in ML features before scaling.")

        
        print("Scaling data for ML models...")
        try:
            scaled_data_ml = scaler.transform(df_ordered_for_scaling)
            print(f"Data scaled successfully, shape: {scaled_data_ml.shape}")
        except Exception as e:
             print(f"Fatal Error during scaling: {e}")
             traceback.print_exc()
             return jsonify({"error": "An error occurred during data scaling."}), 500

        
        print("Running Isolation Forest prediction...")
        iforest_preds = iso_forest.predict(scaled_data_ml)
        ml_flags_df = pd.DataFrame(index=df_processed.index) 
        ml_flags_df['anomaly_iforest'] = (iforest_preds == -1)
        print("Isolation Forest prediction finished.")

        
        print("Running LSTM prediction...")
        ml_flags_df['anomaly_lstm'] = False 
        if len(scaled_data_ml) >= TIMESTEPS:
            print(f"Creating LSTM sequences (timesteps={TIMESTEPS})...")
            sequences = create_sequences(scaled_data_ml, TIMESTEPS)
            if sequences.size > 0 and sequences.ndim == 3 and sequences.shape[2] == len(SCALER_FEATURE_COLS):
                print(f"Predicting with LSTM model on {sequences.shape[0]} sequences...")
                
                sequences_pred = lstm_model.predict(sequences, verbose=0)
                print("Calculating LSTM reconstruction errors...")
                mae_loss = np.mean(np.abs(sequences_pred - sequences), axis=(1, 2))
                anomalous_sequence_indices = np.where(mae_loss > LSTM_MAE_THRESHOLD)[0]
                print(f"Found {len(anomalous_sequence_indices)} anomalous LSTM sequences.")
                for idx in anomalous_sequence_indices:
                    end_original_index = idx + TIMESTEPS - 1
                    start_original_index = idx
                    start_iloc = start_original_index
                    end_iloc = end_original_index + 1
                    if end_iloc <= len(ml_flags_df):
                         ml_flags_df.iloc[start_iloc:end_iloc, ml_flags_df.columns.get_loc('anomaly_lstm')] = True
                    else:
                         print(f"Warning: Index out of bounds mapping LSTM anomaly for sequence ending at {end_original_index}")
            else:
                print(f"No sequences created for LSTM or sequence shape mismatch. Input shape: {scaled_data_ml.shape}, Sequence shape: {sequences.shape}")
        else:
            print(f"Not enough data points ({len(scaled_data_ml)}) to create LSTM sequences with {TIMESTEPS} timesteps.")
        print("LSTM prediction finished.")

        
        print("Combining all anomaly flags...")
        all_flags_df = pd.concat([rule_flags_df, ml_flags_df], axis=1)
        print(f"Flags combined. Shape: {all_flags_df.shape}")

        
        print("Calculating alert levels...")
        alert_levels_df = calculate_alert_levels(all_flags_df)
        print(f"Alert levels calculated. Shape: {alert_levels_df.shape}")

        # --- Email Alert Logic ---
        email_sent = None
        email_status_msg = None
        if 'alert_level_2' in alert_levels_df.columns and alert_levels_df['alert_level_2'].any():
            print("Level 2 alert detected, preparing detailed email...")
            # Find the first instance of a Level 2 alert to detail in the email
            first_level_2_idx = alert_levels_df[alert_levels_df['alert_level_2']].index[0]
            alert_data_row = all_flags_df.loc[first_level_2_idx] # Get the row with all flags for context
            
            # Add original data columns needed for the email body if they exist in df_processed
            for col in ['temperature', 'standardized_COValue'] + [f'standardized_MOX{i}' for i in range(1, 5)]:
                 if col in df_processed.columns:
                      alert_data_row[col] = df_processed.loc[first_level_2_idx, col]
                 else:
                      alert_data_row[col] = None # Ensure the key exists even if column is missing

            # Format timestamp for the specific alert
            if isinstance(first_level_2_idx, pd.Timestamp):
                timestamp_str = first_level_2_idx.strftime('%Y-%m-%d %H:%M:%S')
            else:
                timestamp_str = f"Input index {first_level_2_idx}"

            resident_identifier = "the resident" # Placeholder

            # --- Determine the specific reason(s) for the Level 2 alert ---
            trigger_reasons = []
            critical_rule_triggered = False

            # Check critical rules first
            if alert_data_row.get('anomaly_low_temp', False):
                temp_val = alert_data_row.get('temperature', 'N/A')
                temp_str = f"{temp_val:.1f}°C" if isinstance(temp_val, (int, float)) else temp_val
                trigger_reasons.append(f"Sustained Low Temperature Detected (Value: {temp_str})")
                critical_rule_triggered = True
            if alert_data_row.get('anomaly_high_gas', False):
                co_z = alert_data_row.get('standardized_COValue', None)
                mox_z_max = None
                mox_cols = [f'standardized_MOX{i}' for i in range(1, 5) if f'standardized_MOX{i}' in alert_data_row and alert_data_row[f'standardized_MOX{i}'] is not None]
                if mox_cols:
                   mox_values = [alert_data_row[col] for col in mox_cols]
                   if mox_values:
                       mox_z_max = max(mox_values)

                reason_detail = "High Gas Reading Detected"
                details = []
                if co_z is not None and co_z > 3.5: details.append(f"CO Z-Score: {co_z:.2f}")
                if mox_z_max is not None and mox_z_max > 4.0: details.append(f"Max MOX Z-Score: {mox_z_max:.2f}")
                if details: reason_detail += f" ({', '.join(details)})"
                trigger_reasons.append(reason_detail)
                critical_rule_triggered = True
            if alert_data_row.get('anomaly_inactivity', False):
                trigger_reasons.append("Prolonged Daytime Inactivity Detected")
                critical_rule_triggered = True

            # Check if triggered by multiple flags (if not already triggered by a critical rule)
            all_anomaly_flags = [col for col in alert_data_row.index if col.startswith('anomaly_')]
            anomaly_sum = alert_data_row[all_anomaly_flags].astype(bool).sum() # Sums boolean True as 1

            if not critical_rule_triggered and anomaly_sum >= 2:
                 triggering_flags = [flag for flag in all_anomaly_flags if alert_data_row[flag]]
                 trigger_reasons.append(f"Multiple Concurrent Anomalies Detected: {', '.join(triggering_flags)}")
            elif critical_rule_triggered and anomaly_sum >= 2: # Case where a critical rule AND another flag triggered
                 other_flags = [flag for flag in all_anomaly_flags if alert_data_row[flag] and flag not in ['anomaly_low_temp', 'anomaly_high_gas', 'anomaly_inactivity']]
                 if other_flags:
                      trigger_reasons.append(f"Also detected: {', '.join(other_flags)}")

            # If somehow Level 2 triggered but no specific reason found (shouldn't happen with current logic)
            if not trigger_reasons:
                trigger_reasons.append("Multiple unspecified anomaly indicators met criteria.")

            # --- Construct Email ---
            current_time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') # Get current time
            email_subject = f"URGENT: Level 2 Alert for {resident_identifier}"
            email_body = f"""Dear Caregiver,

This is an urgent Level 2 alert from the Smart Elderly Monitoring System for: {resident_identifier}.

Alert Time: {current_time_str} 

Potential Issue(s) Detected:
"""
            # Add bullet points for each reason
            for reason in trigger_reasons:
                email_body += f"- {reason}\n"

            email_body += f"""
Recommendation:
Please check on the resident's well-being immediately.
You can review the detailed sensor readings and recent activity patterns on the system dashboard.

System Details:
Alert Level: 2 (Critical)
Monitoring System ID: 

If you believe this is a false alarm or require assistance, please contact Support.

Sincerely,
The Smart Elderly Monitoring System
"""

            # Now use email_subject and email_body to send the email
            print("--- Sending Detailed Email ---")
            print(f"Subject: {email_subject}")
            # print("---") # Optional: print body for debugging
            # print(email_body)
            # print("---------------------")
            email_sent, email_status_msg = send_alert_email(email_subject, email_body)

        # --- End Email Alert Logic ---

        
        print("Preparing simplified alert status response...")
        
        alert_status = np.where(alert_levels_df['alert_level_2'], 'Level 2 Triggered',
                        np.where(alert_levels_df['alert_level_1'], 'Level 1 Triggered', 'Normal'))

        
        response_data = []
        for i, idx in enumerate(df_processed.index):
            record = {'status': alert_status[i]}
            if isinstance(idx, pd.Timestamp):
                record['timestamp'] = idx.strftime('%Y-%m-%d %H:%M:%S')
            else:
                record['input_index'] = idx
            
            # Add email status only if a Level 2 alert was triggered for this record
            if alert_levels_df['alert_level_2'].iloc[i] and email_status_msg is not None:
                record['email_status'] = email_status_msg
                
            response_data.append(record)

        
        # Ensure correct order of keys if timestamp/index exists
        final_response_json = []
        for record in response_data:
            ordered_record = {}
            if 'timestamp' in record:
                ordered_record['timestamp'] = record['timestamp']
            elif 'input_index' in record:
                ordered_record['input_index'] = record['input_index']
            ordered_record['status'] = record['status']
            if 'email_status' in record:
                ordered_record['email_status'] = record['email_status']
            final_response_json.append(ordered_record)

        print("Simplified response prepared.")

        print("=== Request Finished ===\n")
        return jsonify(final_response_json)
        return jsonify(response_json)

    except Exception as e:
        print(f"!!! Error during prediction endpoint execution: {e}")
        traceback.print_exc() 
        return jsonify({"error": "An internal error occurred during prediction."}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)

