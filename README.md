# pdf2latex with VLMs

## Dev Environment Setup

### CUDA

Setup development environment:

```sh
conda create -n pdf2latex python=3.11 -y
conda activate pdf2latex
```

Install necessary packages:

```sh
pip install notebook tqdm wandb
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
pip install transformers datasets accelerate
pip install peft
pip install flash-attn --no-build-isolation
```

### MPS

```sh
uv sync
```

Note that `flash-attn` is not available for MPS.

## Cloud Training & Serving on GCP

This repository includes scripts to train and serve the model on Google Cloud Platform (GCP) using Vertex AI.

### Prerequisites

1.  **GCP Project**: You need a Google Cloud Project with billing enabled.
2.  **Terraform**: Install [Terraform](https://developer.hashicorp.com/terraform/downloads) to provision infrastructure.
3.  **Google Cloud SDK**: Install `gcloud` CLI for authentication and submitting jobs.
4.  **Service Account/Auth**: Ensure you are authenticated (`gcloud auth application-default login`) with permissions to create resources.

### 0. Infrastructure Setup (Terraform)

Instead of manually creating resources, you can use Terraform to provision the GCS bucket, Artifact Registry, and enable necessary APIs.

1.  Install [Terraform](https://developer.hashicorp.com/terraform/downloads).
2.  Navigate to the `terraform/` directory.
3.  Update `terraform.tfvars` with your GCP Project ID:
    ```hcl
    project_id = "your-project-id"
    ```
4.  Initialize and Apply:
    ```sh
    terraform init
    terraform apply
    ```

Terraform will output the **Bucket Name** (e.g., `pdf2latex-dataset-xxxx`). Note this down as you will need it for the subsequent steps.

### 1. Upload Dataset

Upload your parquet dataset to your GCS bucket (replace `YOUR_BUCKET_NAME` with the output from Terraform):

```sh
gsutil cp datasets/latex80m_en_1m.parquet gs://YOUR_BUCKET_NAME/datasets/
```

### 2. Build and Push Docker Images

Use the provided script to build the training and serving Docker images and push them to Artifact Registry.

```sh
# Usage: ./scripts/gcp_build_and_push.sh <PROJECT_ID> <REGION> <REPO_NAME>
chmod +x scripts/gcp_build_and_push.sh
./scripts/gcp_build_and_push.sh my-project-id us-central1 pdf2latex-repo
```

This will push:
*   `us-central1-docker.pkg.dev/my-project-id/pdf2latex-repo/pdf2latex-train:latest`
*   `us-central1-docker.pkg.dev/my-project-id/pdf2latex-repo/pdf2latex-serve:latest`

### 3. Submit Training Job

Submit a custom training job to Vertex AI. This uses the `pdf2latex-train` image.

```sh
python scripts/gcp_submit_train.py \
    --project_id my-project-id \
    --location us-central1 \
    --staging_bucket gs://your-bucket-name \
    --display_name pdf2latex-training-job \
    --container_uri us-central1-docker.pkg.dev/my-project-id/pdf2latex-repo/pdf2latex-train:latest \
    --dataset_path gs://your-bucket-name/datasets/latex80m_en_1m.parquet \
    --output_dir gs://your-bucket-name/outputs/run1
```

Monitor the job in the [Vertex AI Training Console](https://console.cloud.google.com/vertex-ai/training/training-pipelines).

### 4. Deploy to Endpoint

Once training is complete, deploy the model to a Vertex AI Endpoint for serving.

```sh
python scripts/gcp_deploy_serve.py \
    --project_id my-project-id \
    --location us-central1 \
    --display_name pdf2latex-endpoint \
    --serving_container_uri us-central1-docker.pkg.dev/my-project-id/pdf2latex-repo/pdf2latex-serve:latest \
    --model_artifact_uri gs://your-bucket-name/outputs/run1/checkpoint-final/ # Adjust path to actual checkpoint
```

### 5. Test Prediction

You can test the deployed endpoint using the Vertex AI SDK or `curl` (if public/authenticated).

```python
from google.cloud import aiplatform

endpoint = aiplatform.Endpoint('projects/my-project-id/locations/us-central1/endpoints/1234567890')
response = endpoint.predict(instances=[...])
```