import argparse
import os
from huggingface_hub import snapshot_download
from google.cloud import storage

def upload_to_gcs(local_path, bucket_name, gcs_prefix, project_id):
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    
    for root, dirs, files in os.walk(local_path):
        for file in files:
            local_file = os.path.join(root, file)
            rel_path = os.path.relpath(local_file, local_path)
            blob_path = os.path.join(gcs_prefix, rel_path)
            
            blob = bucket.blob(blob_path)
            print(f"Uploading {local_file} to gs://{bucket_name}/{blob_path}")
            blob.upload_from_filename(local_file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", required=True, help="Hugging Face Repo ID")
    parser.add_argument("--gcs_uri", required=True, help="Destination GCS URI (gs://bucket/path)")
    parser.add_argument("--project_id", required=True, help="GCP Project ID")
    args = parser.parse_args()
    
    # Download from HF
    print(f"Downloading {args.repo_id} from Hugging Face...")
    local_dir = snapshot_download(repo_id=args.repo_id)
    print(f"Downloaded to {local_dir}")
    
    # Parse GCS URI
    if not args.gcs_uri.startswith("gs://"):
        raise ValueError("gcs_uri must start with gs://")
        
    bucket_name = args.gcs_uri.replace("gs://", "").split("/")[0]
    prefix = "/".join(args.gcs_uri.replace("gs://", "").split("/")[1:])
    
    # Upload to GCS
    print(f"Uploading to {args.gcs_uri}...")
    upload_to_gcs(local_dir, bucket_name, prefix, args.project_id)
    print("Staging complete!")

if __name__ == "__main__":
    main()
