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
    machine_type: str,
    accelerator_type: str,
    accelerator_count: int,
    use_spot: bool,
):
    aiplatform.init(project=project_id, location=location, staging_bucket=staging_bucket)

    job = aiplatform.CustomContainerTrainingJob(
        display_name=display_name,
        container_uri=container_uri,
        # command=["python", "-m", "pdf2latex.train"], # Entrypoint is already set in Dockerfile
    )

    scheduling_strategy = None
    if use_spot:
        scheduling_strategy = aiplatform.compat.types.custom_job.Scheduling.Strategy.SPOT

    training_job = job.run(
        args=[
            f"--dataset_path={dataset_path}",
            f"--output_dir={output_dir}",
        ],
        replica_count=1,
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
        sync=True,
        scheduling_strategy=scheduling_strategy,
    )
    
    return training_job

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit Vertex AI Training Job")
    parser.add_argument("--project_id", required=True, type=str)
    parser.add_argument("--location", default="us-central1", type=str)
    parser.add_argument("--staging_bucket", required=True, type=str)
    parser.add_argument("--display_name", default="pdf2latex-train", type=str)
    parser.add_argument("--container_uri", required=True, type=str)
    parser.add_argument("--dataset_path", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--machine_type", default="n1-standard-4", type=str)
    parser.add_argument("--accelerator_type", default="NVIDIA_TESLA_T4", type=str)
    parser.add_argument("--accelerator_count", default=1, type=int)
    parser.add_argument("--use_spot", action="store_true", help="Use Spot instances for cost savings and quota adherence")
    
    args = parser.parse_args()
    
    submit_training_job(
        project_id=args.project_id,
        location=args.location,
        staging_bucket=args.staging_bucket,
        display_name=args.display_name,
        container_uri=args.container_uri,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        machine_type=args.machine_type,
        accelerator_type=args.accelerator_type,
        accelerator_count=args.accelerator_count,
        use_spot=args.use_spot,
    )
