import argparse
import base64
from google.cloud import aiplatform
from google.protobuf import json_format
from google.cloud.aiplatform.gapic.schema import predict

def predict_image(project_id, location, endpoint_id, image_path):
    aiplatform.init(project=project_id, location=location)

    endpoint = aiplatform.Endpoint(endpoint_id)

    with open(image_path, "rb") as f:
        file_content = f.read()

    # The format depends on how vLLM expects the input.
    # For Qwen2-VL, it usually expects a chat template structure.
    # We need to construct the prompt matching the serving container's expectation.
    # vLLM's /v1/completions or /v1/chat/completions API is standard.
    # Vertex AI sends the "instances" list as the body.
    
    # However, when deploying vLLM on Vertex AI with a custom container, 
    # the input format is determined by how vLLM handles the Vertex AI payload.
    # Usually, Vertex AI sends: {"instances": [...], "parameters": {...}}
    # vLLM might expect standard OpenAI format if we used the OpenAI serving entrypoint,
    # but Vertex AI wraps it.
    
    # Let's try the standard Vertex AI prediction format for custom containers.
    # If vLLM is running as an OpenAI server, we might need a proxy or specific payload.
    # But usually, for custom containers, we just send what the container expects.
    
    # Qwen2-VL via vLLM OpenAI API expects Chat Completions format
    encoded_image = base64.b64encode(file_content).decode("utf-8")
    
    # Construct OpenAI-compatible payload
    # Model name is '/model-artifacts' because that's what we passed to vLLM
    payload = {
        "model": "/model-artifacts",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Convert this to LaTeX."},
                    {
                        "type": "image_url", 
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 512,
        "temperature": 0.2
    }
    
    print("Sending request to /v1/chat/completions...")
    
    # Use raw_predict to bypass Vertex AI's "instances" wrapper
    # This sends the JSON body directly to the container
    import json
    response = endpoint.raw_predict(
        body=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    
    print("Prediction Response:")
    # Response is bytes, decode it
    print(response.content.decode("utf-8"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_id", required=True)
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--endpoint_id", required=True)
    parser.add_argument("--image_path", required=True)
    args = parser.parse_args()

    predict_image(args.project_id, args.location, args.endpoint_id, args.image_path)
