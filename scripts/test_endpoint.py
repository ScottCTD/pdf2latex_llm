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
    
    # Qwen2-VL expects:
    # <|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Convert this to LaTeX.<|im_end|>\n<|im_start|>assistant\n
    
    encoded_image = base64.b64encode(file_content).decode("utf-8")
    
    # We will construct a prompt that vLLM can process.
    # Since we are using the raw vLLM serve, it exposes an OpenAI-compatible API.
    # Vertex AI's predict() method sends a POST request to the container's predict route.
    # In our deploy script, we set `serving_container_predict_route="/v1/completions"`.
    # So we need to send a payload that matches the OpenAI Completions API.
    
    prompt = f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Convert this to LaTeX.<|im_end|>\n<|im_start|>assistant\n"
    
    # vLLM with Qwen2-VL supports passing image data in the prompt or as separate arguments depending on the version.
    # A common way for vLLM serving multimodal models via API is somewhat complex.
    # Ideally, we should use the /v1/chat/completions endpoint if possible, but we set /v1/completions.
    
    # Let's try a simple completion payload.
    # Note: Passing images to vLLM via the standard HTTP API can be tricky if it doesn't support base64 directly in the prompt.
    # But let's assume standard OpenAI format for now.
    
    instance = {
        "prompt": prompt,
        "max_tokens": 512,
        "temperature": 0.2,
        "image_data": [{"data": encoded_image, "media_type": "image/png"}] # This is a guess on vLLM's custom input extension
    }
    
    # Actually, vLLM's OpenAI server might not support "image_data" in /v1/completions directly in this way.
    # It's safer to use the Chat Completions API if we can, but we bound to /v1/completions.
    
    # Alternative: We can try to use the raw predict request if we knew the signature.
    # Given the uncertainty, let's try the most standard "prompt" approach.
    
    response = endpoint.predict(instances=[instance])
    
    print("Prediction Response:")
    print(response)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_id", required=True)
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--endpoint_id", required=True)
    parser.add_argument("--image_path", required=True)
    args = parser.parse_args()

    predict_image(args.project_id, args.location, args.endpoint_id, args.image_path)
