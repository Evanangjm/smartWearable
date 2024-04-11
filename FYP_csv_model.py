#!/usr/bin/env python
# coding: utf-8

# # Running on Flask

# In[ ]:


import serial
import joblib
import numpy as np
import pandas as pd

# Load the trained models and scaler
scaler = joblib.load('scaler.gz')
rf_classifier_e = joblib.load('rf_emotion_model.gz')
rf_classifier_s = joblib.load('rf_stress_model.gz')

# Serial port configuration
SERIAL_PORT = '/dev/cu.usbserial-56430171891'
BAUD_RATE = 115200
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

# Correct feature names based on the differences being calculated
feature_names = ['TempDiff', 'HumidityDiff', 'HeartRateDiff', 'OxygenDiff', 'GSRDiff', 'BodyTempDiff']
sensor_labels = ['Temperature', 'Humidity', 'Heart Rate', 'Oxygen', 'GSR', 'Body Temperature']

def predict_emotion_stress(values, feature_names, sensor_labels, scalar, rf_classifier_e, rf_classifier_s):
    # Extracting sensor values and baseline values based on the provided sequence
    sensor_values = values[:6]  # Includes rawGsrValue at the end
    baseline_values = [values[8], values[9], values[10], values[11], values[12], values[13]]

    # Calculating differences between sensor readings and baseline values
    data_diff = [
        sensor_values[0] - baseline_values[1],  # Temperature - envTempBaseline
        sensor_values[1] - baseline_values[2],  # Humidity - humidityBaseline
        sensor_values[2] - baseline_values[3],  # HeartRate - heartRateBaseline
        sensor_values[3] - baseline_values[4],  # Oxygen - oxygenBaseline
        sensor_values[5] - baseline_values[0],  # rawGsrValue - gsrBaseline
        sensor_values[4] - baseline_values[5],  # BodyTemp - tempBaseline
    ]

    df_diff = pd.DataFrame([data_diff], columns=feature_names)
    scaled_data = scalar.transform(df_diff)
    emotion_pred = rf_classifier_e.predict(scaled_data)
    stress_pred = rf_classifier_s.predict(scaled_data)

    # Print Predicted Emotional State and Stress Level along with details
    print(f"\nPredicted Emotional State: {emotion_pred[0]}, Predicted Stress Level: {stress_pred[0]}")
    for label, diff in zip(sensor_labels, data_diff):
        print(f"{label} Difference: {diff:.2f}")




# In[ ]:


def connect_and_predict():
# Load the trained models and scaler
    scaler = joblib.load('scaler.gz')
    rf_classifier_e = joblib.load('rf_emotion_model.gz')
    rf_classifier_s = joblib.load('rf_stress_model.gz')

    # Serial port configuration
    SERIAL_PORT = '/dev/cu.usbserial-56430171891'
    BAUD_RATE = 115200
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

    # Correct feature names based on the differences being calculated
    feature_names = ['TempDiff', 'HumidityDiff', 'HeartRateDiff', 'OxygenDiff', 'GSRDiff', 'BodyTempDiff']
    sensor_labels = ['Temperature', 'Humidity', 'Heart Rate', 'Oxygen', 'GSR', 'Body Temperature']

    try:
        print("Starting real-time data prediction...")
        while True:
            if ser.in_waiting > 0:
                serial_line = ser.readline().decode('utf-8').strip()
                values = list(map(float, serial_line.split(',')))
                if len(values) == 14:  # Ensure correct number of values
                    predict_emotion_stress(values, feature_names, sensor_labels, scalar=scaler, rf_classifier_e=rf_classifier_e, rf_classifier_s=rf_classifier_s)
                else:
                    print(f"Incorrect number of values received: {len(values)}")
    except KeyboardInterrupt:
        print("\nReal-time data prediction stopped by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        ser.close()
        print("Serial connection closed.")

