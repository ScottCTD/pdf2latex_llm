import argparse
from google.cloud import aiplatform

def deploy_model(
    project_id: str,
    location: str,
    display_name: str,
    serving_container_uri: str,
    model_artifact_uri: str,
    machine_type: str = "g2-standard-4",
):
    aiplatform.init(project=project_id, location=location)

    # 1. Upload Model to Registry
    model = aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri=model_artifact_uri,
        serving_container_image_uri=serving_container_uri,
        serving_container_args=["--model", "/model-artifacts"], # Adjust based on where artifacts are mounted/downloaded
        # vLLM specific args can be passed here or via env vars
        serving_container_ports=[8000],
        serving_container_predict_route="/v1/completions",
        serving_container_health_route="/health",
    )
    
    print(f"Model uploaded: {model.resource_name}")

    # 2. Create Endpoint
    endpoint = aiplatform.Endpoint.create(
        display_name=f"{display_name}-endpoint",
    )
    print(f"Endpoint created: {endpoint.resource_name}")

    # 3. Deploy Model to Endpoint
    model.deploy(
        endpoint=endpoint,
        machine_type=machine_type,
        accelerator_type="NVIDIA_L4",
        accelerator_count=1,
        min_replica_count=1,
        max_replica_count=1,
        sync=False,
    )
    
    print(f"Model deployment initiated to endpoint: {endpoint.resource_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy model to Vertex AI")
    parser.add_argument("--project_id", required=True, type=str)
    parser.add_argument("--location", default="us-central1", type=str)
    parser.add_argument("--display_name", default="pdf2latex-model", type=str)
    parser.add_argument("--serving_container_uri", required=True, type=str)
    parser.add_argument("--model_artifact_uri", required=True, type=str, help="GCS path to model artifacts")
    
    args = parser.parse_args()
    
    deploy_model(
        project_id=args.project_id,
        location=args.location,
        display_name=args.display_name,
        serving_container_uri=args.serving_container_uri,
        model_artifact_uri=args.model_artifact_uri,
    )
