import os
import subprocess
import sys
from google.cloud import storage

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

def download_gcs_folder(gcs_uri, local_dir):
    """Downloads a folder from GCS to a local directory."""
    if not gcs_uri.startswith("gs://"):
        print(f"Invalid GCS URI: {gcs_uri}")
        return

    bucket_name = gcs_uri.replace("gs://", "").split("/")[0]
    prefix = "/".join(gcs_uri.replace("gs://", "").split("/")[1:])
    
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    print(f"Downloading from GCS: {gcs_uri} to {local_dir}")
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)

    for blob in blobs:
        if blob.name.endswith("/"):
            continue
            
        # Remove prefix from blob name to get relative path
        rel_path = blob.name[len(prefix):] if prefix else blob.name
        local_path = os.path.join(local_dir, rel_path)
        
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        print(f"Downloading {blob.name} to {local_path}")
        blob.download_to_filename(local_path)

def main():
    # Check for manual GCS URI first, then Vertex AI managed URI
    gcs_uri = os.environ.get("MODEL_GCS_URI") or os.environ.get("AIP_STORAGE_URI")
    model_dir = "/model-artifacts"

    if gcs_uri:
        print(f"Found AIP_STORAGE_URI: {gcs_uri}")
        try:
            download_gcs_folder(gcs_uri, model_dir)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading from GCS: {e}")
            # We don't exit here, we let vLLM try to run. 
            # If the user passed a HF model ID in args, it might still work.
    else:
        print("No AIP_STORAGE_URI found. Assuming Hugging Face model ID is passed in args.")

    # Construct the vllm command
    # We pass through all arguments sent to the container
    cmd = ["vllm", "serve"] + sys.argv[1:]
    
    print(f"Starting vLLM: {' '.join(cmd)}")
    subprocess.check_call(cmd)

if __name__ == "__main__":
    main()
