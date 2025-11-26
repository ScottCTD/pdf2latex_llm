import argparse
from google.cloud import aiplatform

def deploy_model(
    project_id: str,
    location: str,
    display_name: str,
    serving_container_uri: str,
    machine_type: str,
    accelerator_type: str,
    accelerator_count: int,
    model_artifact_uri: str = None,
    hf_model_id: str = None,
    service_account: str = None,
):
    aiplatform.init(project=project_id, location=location)

    if hf_model_id:
        serving_args = ["--model", hf_model_id]
        # If using HF model, we might not have a GCS artifact URI
        # But Model.upload might require it or we can pass None if allowed, 
        # or we just don't pass artifact_uri.
        # However, Vertex AI Model registry usually expects some artifact or container.
        # If we only provide container, it's fine.
    else:
        serving_args = ["--model", "/model-artifacts"]
    # The original serving_args logic is now moved to serving_container_args in Model.upload
    # and the deploy args are moved there as well.
    # We will use a fixed set of args for the serving container.
    # The `serving_args` variable is no longer needed for Model.upload.

    # 1. Upload Model to Registry
    # We pass the GCS URI as an env var to avoid Vertex AI copying artifacts to a managed bucket,
    # which seems to be causing missing file issues.
    env_vars = {"MODEL_GCS_URI": model_artifact_uri} if model_artifact_uri else {}
    
    model = aiplatform.Model.upload(
        display_name=display_name,
        serving_container_image_uri=serving_container_uri,
        serving_container_environment_variables=env_vars,
        # Pass V100-compatible args here, PLUS the model path
        serving_container_args=serving_args + ["--dtype", "float16", "--max-model-len", "8192", "--enforce-eager"],
        serving_container_ports=[8000],
        serving_container_predict_route="/v1/chat/completions",
        serving_container_health_route="/health",
    )
    
    print(f"Model uploaded: {model.resource_name}")

    # 2. Get or Create Endpoint
    endpoints = aiplatform.Endpoint.list(filter=f"display_name={display_name}-endpoint")
    if endpoints:
        endpoint = endpoints[0]
        print(f"Found existing endpoint: {endpoint.resource_name}")
        
        # Undeploy all existing models to free up quota
        print("Undeploying existing models to free up quota...")
        endpoint.undeploy_all(sync=True)
        print("All models undeployed.")
    else:
        endpoint = aiplatform.Endpoint.create(
            display_name=f"{display_name}-endpoint",
        )
        print(f"Endpoint created: {endpoint.resource_name}")

    # 3. Deploy Model to Endpoint
    print(f"Deploying model to Endpoint : {endpoint.resource_name}")
    
    # Generate Log URL
    log_url = f"https://console.cloud.google.com/logs/viewer?project={project_id}&resource=aiplatform.googleapis.com%2FEndpoint%2Fendpoint_id%2F{endpoint.name.split('/')[-1]}"
    print(f"\n📢 MONITOR DEPLOYMENT LOGS HERE: {log_url}\n")

    model.deploy(
        endpoint=endpoint,
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
        min_replica_count=1,
        max_replica_count=1,
        sync=True,
        # V100 requires float16 (no bfloat16 support) and we need to be careful with attention
        service_account=service_account, 
        # The 'args' parameter is removed as per instruction.
    )
    
    print(f"Model deployment initiated to endpoint: {endpoint.resource_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy model to Vertex AI")
    parser.add_argument("--project_id", required=True, type=str)
    parser.add_argument("--location", default="us-central1", type=str)
    parser.add_argument("--display_name", default="pdf2latex-model", type=str)
    parser.add_argument("--serving_container_uri", required=True, type=str)
    parser.add_argument("--model_artifact_uri", type=str, help="GCS path to model artifacts")
    parser.add_argument("--hf_model_id", type=str, help="Hugging Face Model ID")
    parser.add_argument("--machine_type", default="n1-standard-4", type=str)
    parser.add_argument("--accelerator_type", default="NVIDIA_TESLA_V100", type=str)
    parser.add_argument("--accelerator_count", default=1, type=int)
    parser.add_argument("--service_account", type=str, help="Service account for the container")
    
    args = parser.parse_args()
    
    if not args.model_artifact_uri and not args.hf_model_id:
        parser.error("One of --model_artifact_uri or --hf_model_id must be provided.")

    deploy_model(
        project_id=args.project_id,
        location=args.location,
        display_name=args.display_name,
        serving_container_uri=args.serving_container_uri,
        model_artifact_uri=args.model_artifact_uri,
        hf_model_id=args.hf_model_id,
        machine_type=args.machine_type,
        accelerator_type=args.accelerator_type,
        accelerator_count=args.accelerator_count,
        service_account=args.service_account,
    )
