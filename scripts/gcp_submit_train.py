import argparse
from google.cloud import aiplatform

def submit_training_job(
    project_id: str,
    location: str,
    staging_bucket: str,
    display_name: str,
    container_uri: str,
    dataset_path: str,
    output_dir: str,
    machine_type: str = "g2-standard-4",
    accelerator_type: str = "NVIDIA_L4",
    accelerator_count: int = 1,
):
    aiplatform.init(project=project_id, location=location, staging_bucket=staging_bucket)

    job = aiplatform.CustomContainerTrainingJob(
        display_name=display_name,
        container_uri=container_uri,
        # command=["python", "-m", "pdf2latex.train"], # Entrypoint is already set in Dockerfile
    )

    model = job.run(
        args=[
            f"--dataset_path={dataset_path}",
            f"--output_dir={output_dir}",
        ],
        replica_count=1,
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
        sync=False,
    )
    
    print(f"Training job submitted. Resource name: {job.resource_name}")
    return job

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit Vertex AI Training Job")
    parser.add_argument("--project_id", required=True, type=str)
    parser.add_argument("--location", default="us-central1", type=str)
    parser.add_argument("--staging_bucket", required=True, type=str)
    parser.add_argument("--display_name", default="pdf2latex-train", type=str)
    parser.add_argument("--container_uri", required=True, type=str)
    parser.add_argument("--dataset_path", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    
    args = parser.parse_args()
    
    submit_training_job(
        project_id=args.project_id,
        location=args.location,
        staging_bucket=args.staging_bucket,
        display_name=args.display_name,
        container_uri=args.container_uri,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
    )
