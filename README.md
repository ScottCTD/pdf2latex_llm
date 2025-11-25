# pdf2latex with VLMs

This repository contains a complete MLOps pipeline for training and serving a PDF-to-LaTeX model on Google Cloud Platform (GCP) using Vertex AI.

## 🛠️ Local Development Setup

### CUDA (Linux/Windows)
```sh
conda create -n pdf2latex python=3.11 -y
conda activate pdf2latex
pip install notebook tqdm wandb
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
pip install transformers datasets accelerate peft flash-attn --no-build-isolation
```

### MPS (Mac)
```sh
uv sync
```
*Note: `flash-attn` is not available for MPS.*

---

## ☁️ GCP MLOps Pipeline

### Prerequisites
1.  **GCP Project**: A Google Cloud Project with billing enabled.
2.  **Tools**: Install [Terraform](https://developer.hashicorp.com/terraform/downloads), [Google Cloud SDK](https://cloud.google.com/sdk/docs/install), and [uv](https://github.com/astral-sh/uv).
3.  **Authentication**:
    ```sh
    gcloud auth login
    gcloud auth application-default login
    ```

### 1. Infrastructure Setup (Terraform)
Provision all necessary resources (GCS Bucket, Artifact Registry, APIs) automatically.

```sh
cd terraform
# Update terraform.tfvars with your project_id
terraform init
terraform apply
```
*Note down the **Bucket Name** output by Terraform.*

### 2. Dataset & Model Staging
Generate the dataset and stage the model artifacts to GCS.

**Generate Dataset:**
```sh
uv run python pdf2latex/data_process.py
# Upload to GCS
gcloud storage cp datasets/latex80m_en_1m.parquet gs://YOUR_BUCKET/datasets/
```

**Stage Model (Hugging Face -> GCS):**
Download the model and upload it to your bucket for controlled serving.
```sh
uv run python scripts/stage_model.py \
    --repo_id scottcfy/Qwen2-VL-2B-Instruct-pdf2latex \
    --gcs_uri gs://YOUR_BUCKET/models/pdf2latex-v1 \
    --project_id YOUR_PROJECT_ID
```

### 3. Build & Push Docker Images
Build the training and serving containers and push them to Artifact Registry.

```sh
# Usage: ./scripts/gcp_build_and_push.sh <PROJECT_ID> <REGION> <REPO_NAME>
./scripts/gcp_build_and_push.sh YOUR_PROJECT_ID us-central1 pdf2latex-repo
```

### 4. Training (Optional)
Submit a custom training job to Vertex AI.

```sh
uv run python scripts/gcp_submit_train.py \
    --project_id YOUR_PROJECT_ID \
    --location us-central1 \
    --staging_bucket gs://YOUR_BUCKET \
    --display_name pdf2latex-train \
    --container_uri us-central1-docker.pkg.dev/YOUR_PROJECT_ID/pdf2latex-repo/pdf2latex-train:latest \
    --dataset_path gs://YOUR_BUCKET/datasets/latex80m_en_1m.parquet \
    --output_dir gs://YOUR_BUCKET/outputs/run1 \
    --use_spot  # Use Spot instances for cost savings
```

### 5. Serving / Deployment
Deploy the model to a Vertex AI Endpoint. The serving container supports loading from GCS or Hugging Face.

**Deploy from GCS (Recommended):**
```sh
uv run python scripts/gcp_deploy_serve.py \
    --project_id YOUR_PROJECT_ID \
    --location us-central1 \
    --display_name pdf2latex-serve \
    --serving_container_uri us-central1-docker.pkg.dev/YOUR_PROJECT_ID/pdf2latex-repo/pdf2latex-serve:latest \
    --model_artifact_uri gs://YOUR_BUCKET/models/pdf2latex-v1
```

**Deploy from Hugging Face directly:**
```sh
uv run python scripts/gcp_deploy_serve.py \
    ...
    --hf_model_id scottcfy/Qwen2-VL-2B-Instruct-pdf2latex
```

### 6. Testing
Verify the deployed endpoint by sending a sample image.

```sh
uv run python scripts/test_endpoint.py \
    --project_id YOUR_PROJECT_ID \
    --endpoint_id YOUR_ENDPOINT_ID \
    --image_path test_image.png
```