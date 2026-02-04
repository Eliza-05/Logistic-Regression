import time
import sagemaker
from sagemaker.sklearn.model import SKLearnModel

print("="*70)
print("SAGEMAKER DEPLOYMENT - Heart Disease Prediction Model")
print("="*70)

print("\n📋 Step 1: Setting up SageMaker environment...")
session = sagemaker.Session()
region = session.boto_region_name
bucket = session.default_bucket()
role = sagemaker.get_execution_role()

print(f"✅ Region: {region}")
print(f"✅ Role: {role}")
print(f"✅ Bucket: {bucket}")

print("\n📋 Step 2: Using model tarball from S3...")
model_data = f"s3://{bucket}/heart-disease-model/model.tar.gz"
print(f"✅ model_data: {model_data}")

print("\n📋 Step 3: Creating SageMaker model...")
model = SKLearnModel(
    model_data=model_data,
    role=role,
    entry_point="inference.py",
    source_dir="model_artifacts",
    framework_version="1.2-1",
    py_version="py3",
    sagemaker_session=session
)
print("✅ Model object ready")

print("\n📋 Step 4: Deploying endpoint...")
endpoint_name = f"heart-disease-endpoint-{int(time.time())}"

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.t2.medium",   
    endpoint_name=endpoint_name
)

print("\n" + "="*70)
print("🎉 DEPLOYMENT SUCCESSFUL!")
print("="*70)
print(f"Endpoint Name: {endpoint_name}")
