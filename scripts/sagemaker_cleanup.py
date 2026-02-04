"""
sagemaker_cleanup.py

Ejecuta este script cuando termines para ELIMINAR el endpoint y evitar cobros.
"""

import boto3
import sagemaker

print("="*70)
print("SAGEMAKER RESOURCE CLEANUP")
print("="*70)

endpoint_name = 'heart-disease-endpoint'


sagemaker_client = boto3.client('sagemaker')
session = sagemaker.Session()

print(f"\nDeleting endpoint: {endpoint_name}...")

try:
 
    sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
    print("Endpoint deleted successfully!")
    
    
    sagemaker_client.delete_endpoint_config(
        EndpointConfigName=endpoint_name
    )
    print("Endpoint configuration deleted!")
    
    print("\n" + "="*70)
    print("CLEANUP COMPLETED")
    print("="*70)
    print("\nAll billable resources have been deleted.")
    print("\nNote: Model artifacts in S3 remain (minimal cost).")
    print("To delete S3 data, go to S3 console and delete the bucket.")
    
except Exception as e:
    print(f"Error during cleanup: {e}")
    print("\nManual cleanup:")
    print("1. Go to SageMaker Console")
    print("2. Navigate to Endpoints")
    print("3. Delete the endpoint manually")