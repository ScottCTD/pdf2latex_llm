from huggingface_hub import HfApi, HfFolder, Repository, create_repo
from huggingface_hub import upload_folder

# 1. Create repo (public or private)
create_repo("scottcfy/Qwen2-VL-2B-Instruct-pdf2latex", private=False)

# 2. Upload folder
upload_folder(
    folder_path="./outputs/1/checkpoint-6000/merged",      # path to your saved model
    repo_id="scottcfy/Qwen2-VL-2B-Instruct-pdf2latex",  # repo name on Hugging Face Hub
    commit_message="Initial model upload 🚀"
)
