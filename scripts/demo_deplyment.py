import argparse
import json
import numpy as np
import os

try:
    import sagemaker
    from sagemaker.sklearn.model import SKLearnModel
except Exception:
    sagemaker = None
    SKLearnModel = None


def print_header():
    print("=" * 60)
    print("HEART DISEASE PREDICTION - DEPLOYMENT DEMO")
    print("=" * 60)
    print("Flujo de despliegue y prueba local.")



def setup_sagemaker():
    print("Paso 1: Inicializar sesion de SageMaker")
    if sagemaker is None:
        print("   - SageMaker no disponible. Continuando sin sesion real.")
        return None, "simulated-bucket", "simulated-role"
    try:
        session = sagemaker.Session()
        bucket = session.default_bucket()
        role = sagemaker.get_execution_role()
        print(f"   - Bucket: {bucket}")
        print(f"   - Role: {role[:40]}...")
        return session, bucket, role
    except Exception as exc:
        print(f"   - No se pudo crear sesion real: {exc}")
        print("   - Continuando sin sesion real.")
        return None, "simulated-bucket", "simulated-role"


def upload_model(session, bucket):
    print("Paso 2: Subir modelo a S3")
    if session is None:
        s3_path = f"s3://{bucket}/heart-disease-model/model_artifacts.tar.gz"
        print(f"   - Modelo en S3: {s3_path}")
        return s3_path
    if not os.path.exists("model_artifacts.tar.gz"):
        s3_path = f"s3://{bucket}/heart-disease-model/model_artifacts.tar.gz"
        print("   - No existe model_artifacts.tar.gz local. Usando ruta base.")
        print(f"   - Modelo en S3: {s3_path}")
        return s3_path
    s3_path = session.upload_data(
        path="model_artifacts.tar.gz",
        bucket=bucket,
        key_prefix="heart-disease-model",
    )
    print(f"   - Modelo en S3: {s3_path}")
    return s3_path


def create_model_obj(s3_path, role, session):
    print("Paso 3: Crear objeto SKLearnModel")
    if SKLearnModel is None or session is None:
        print("   - No se creo objeto real.")
        return None
    model = SKLearnModel(
        model_data=s3_path,
        role=role,
        entry_point="inference.py",
        source_dir="model_artifacts",
        framework_version="1.2-1",
        py_version="py3",
        sagemaker_session=session,
    )
    print("   - Objeto modelo creado.")
    return model


def show_config(s3_path):
    print("\nPaso 4: Configuracion de despliegue")
    print(f"   Modelo: {s3_path}")
    print("   Tipo de instancia: ml.t2.medium")
    print("   Nombre de endpoint: heart-disease-demo-endpoint")
    print("   Despliegue real omitido por restriccion de permisos.\n")


def local_inference():
    print("Paso 5: Prueba de inferencia local")
    weights = np.load("model_artifacts/weights.npy")
    bias = np.load("model_artifacts/bias.npy")
    norm = np.load(
        "model_artifacts/normalization_params.npy",
        allow_pickle=True,
    ).item()
    meta = np.load(
        "model_artifacts/model_metadata.npy",
        allow_pickle=True,
    ).item()

    mu = norm["mu"]
    sigma = norm["sigma"]
    features = meta["feature_names"]

    # Ejemplo de pacientes (orden igual a feature_names)
    pacientes = [
        {
            "nombre": "Paciente A",
            "datos": [60, 140, 250, 150, 1.2, 1, 1],
        },
        {
            "nombre": "Paciente B",
            "datos": [45, 120, 180, 170, 0.0, 0, 0],
        },
    ]
    for p in pacientes:
        x = np.array(p["datos"])
        if x.shape[0] != len(features):
            raise ValueError(
                f"Datos de {p['nombre']} con longitud {x.shape[0]}, "
                f"pero se esperan {len(features)} features."
            )
        x_norm = (x - mu) / sigma
        z = np.dot(x_norm, weights) + bias
        prob = 1 / (1 + np.exp(-z))
        resultado = "Enfermedad cardiaca" if prob >= 0.5 else "Sin enfermedad"
        print(f"   {p['nombre']}: Probabilidad={prob:.2%} -> {resultado}")


def ensure_sample_json(path, features):
    if os.path.exists(path):
        return
    sample = {
        "features_order": features,
        "patients": [
            {
                "name": "Paciente A",
                "features": [60, 140, 250, 150, 1.2, 1, 1],
            },
            {
                "name": "Paciente B",
                "features": [45, 120, 180, 170, 0.0, 0, 0],
            },
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)


def inference_from_json(input_path, output_path):
    print("Paso 6: Inferencia desde JSON")
    weights = np.load("model_artifacts/weights.npy")
    bias = np.load("model_artifacts/bias.npy")
    norm = np.load(
        "model_artifacts/normalization_params.npy",
        allow_pickle=True,
    ).item()
    meta = np.load(
        "model_artifacts/model_metadata.npy",
        allow_pickle=True,
    ).item()

    mu = norm["mu"]
    sigma = norm["sigma"]
    features = meta["feature_names"]

    ensure_sample_json(input_path, features)
    with open(input_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if payload.get("features_order") != features:
        raise ValueError(
            "El orden de features en el JSON no coincide con el del modelo. "
            f"Esperado: {features}"
        )

    results = []
    for item in payload.get("patients", []):
        name = item.get("name", "Paciente")
        x = np.array(item["features"])
        if x.shape[0] != len(features):
            raise ValueError(
                f"Datos de {name} con longitud {x.shape[0]}, "
                f"pero se esperan {len(features)} features."
            )
        x_norm = (x - mu) / sigma
        z = np.dot(x_norm, weights) + bias
        prob = 1 / (1 + np.exp(-z))
        prediction = 1 if prob >= 0.5 else 0
        results.append(
            {
                "name": name,
                "prediction": int(prediction),
                "probability": float(prob),
                "risk_level": "HIGH RISK (Disease)" if prediction == 1 else "LOW RISK (Healthy)",
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2, ensure_ascii=False)

    print(f"   - Entrada: {input_path}")
    print(f"   - Salida: {output_path}")
    for r in results:
        print(f"   {r['name']}: Prob={r['probability']:.2%} -> {r['risk_level']}")


def resumen():
    print("\nDEMO COMPLETADA.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo deployment sin endpoint real.")
    parser.add_argument(
        "--input-json",
        default="demo_request.json",
        help="Ruta del JSON de entrada con features.",
    )
    parser.add_argument(
        "--output-json",
        default="demo_response.json",
        help="Ruta del JSON de salida con predicciones.",
    )
    parser.add_argument(
        "--skip-local",
        action="store_true",
        help="Omite la inferencia local de ejemplo.",
    )
    args = parser.parse_args()

    print_header()
    session, bucket, role = setup_sagemaker()
    s3_path = upload_model(session, bucket)
    model = create_model_obj(s3_path, role, session)
    show_config(s3_path)
    if not args.skip_local:
        local_inference()
    inference_from_json(args.input_json, args.output_json)
    resumen()