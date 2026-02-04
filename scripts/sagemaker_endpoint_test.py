"""
sagemaker_endpoint_test.py

Ejecuta esto en tu notebook de SageMaker DESPUÉS de que el endpoint esté desplegado.
"""

import json
import boto3
import sagemaker
from sagemaker.predictor import Predictor

print("="*70)
print("TESTING SAGEMAKER ENDPOINT")
print("="*70)


endpoint_name = 'heart-disease-endpoint'
predictor = Predictor(endpoint_name=endpoint_name)

print(f"\nConnected to endpoint: {endpoint_name}")


print("\n" + "="*70)
print("TEST 1: HIGH RISK PATIENT")
print("="*70)

high_risk_patient = {
    "Age": 65,
    "BP": 160,
    "Cholesterol": 300,
    "Max HR": 110,
    "ST depression": 3.5,
    "Number of vessels fluro": 3,
    "Exercise angina": 1
}

print("\nInput:")
print(json.dumps(high_risk_patient, indent=2))

response = predictor.predict(high_risk_patient)
print("\nPrediction:")
print(json.dumps(response, indent=2))


print("\n" + "="*70)
print("TEST 2: LOW RISK PATIENT")
print("="*70)

low_risk_patient = {
    "Age": 35,
    "BP": 120,
    "Cholesterol": 180,
    "Max HR": 170,
    "ST depression": 0.0,
    "Number of vessels fluro": 0,
    "Exercise angina": 0
}

print("\nInput:")
print(json.dumps(low_risk_patient, indent=2))

response = predictor.predict(low_risk_patient)
print("\nPrediction:")
print(json.dumps(response, indent=2))


print("\n" + "="*70)
print("TEST 3: BATCH PREDICTIONS")
print("="*70)

test_patients = [
    {"Age": 50, "BP": 140, "Cholesterol": 250, "Max HR": 140, "ST depression": 1.5, "Number of vessels fluro": 1, "Exercise angina": 0},
    {"Age": 60, "BP": 150, "Cholesterol": 280, "Max HR": 120, "ST depression": 2.0, "Number of vessels fluro": 2, "Exercise angina": 1},
    {"Age": 40, "BP": 130, "Cholesterol": 200, "Max HR": 160, "ST depression": 0.5, "Number of vessels fluro": 0, "Exercise angina": 0}
]

for i, patient in enumerate(test_patients, 1):
    response = predictor.predict(patient)
    print(f"\nPatient {i}: {response['risk_level']} (p={response['probability']:.2%})")


print("\n" + "="*70)
print("ENDPOINT METRICS")
print("="*70)

cloudwatch = boto3.client('cloudwatch')
print(f"\nEndpoint: {endpoint_name}")
print("\nAvailable metrics in CloudWatch:")
print("  • ModelLatency")
print("  • Invocations")
print("  • InvocationErrors")
print("\nAccess full metrics at:")
print("AWS Console → CloudWatch → Metrics → SageMaker")

print("\n" + "="*70)
print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
print("="*70)