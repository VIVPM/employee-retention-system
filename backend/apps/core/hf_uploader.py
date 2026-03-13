import os
from huggingface_hub import HfApi, create_repo
import dotenv

class HFUploader:
    def __init__(self, logger=None):
        self.logger = logger
        dotenv.load_dotenv()
        self.token = os.getenv("HF_TOKEN")
        self.repo_id = os.getenv("HF_REPO_ID", "")
        
        if not self.token:
            if self.logger:
                self.logger.info("HF_TOKEN not found in .env, skipping upload.")
            return
            
        self.api = HfApi(token=self.token)

    def upload_models(self):
        """Uploads the apps/models directory directly to the active Hugging Face model repository."""
        if not self.token:
            return False
            
        # Ensure repo exists just before upload
        try:
            self.api.repo_info(repo_id=self.repo_id, repo_type="model")
        except Exception:
            if self.logger:
                self.logger.info(f"Creating new repo {self.repo_id} on Hugging Face Hub.")
            create_repo(repo_id=self.repo_id, token=self.token, repo_type="model", exist_ok=True, private=False)

        if self.logger:
            self.logger.info(f"Uploading models to {self.repo_id}...")
            
        try:
            # Upload the entire apps/models directory (model subfolders + results.csv)
            # Determine next version
            try:
                refs = self.api.list_repo_refs(repo_id=self.repo_id)
                tags = [tag.name for tag in refs.tags if tag.name.startswith("v")]
                if tags:
                    # Sort version tags and pick the latest
                    latest_tag = sorted(tags, key=lambda x: float(x.replace("v", "")))[-1]
                    latest_version_num = float(latest_tag.replace("v", ""))
                    new_version = f"v{latest_version_num + 1.0}"
                else:
                    new_version = "v1.0"
            except:
                new_version = "v1.0"

            if os.path.exists("apps/models"):
                self.api.upload_folder(
                    repo_id=self.repo_id,
                    folder_path="apps/models",
                    path_in_repo="models",
                    commit_message=f"Update trained models and results - {new_version}"
                )
                
                # Tag the release
                self.api.create_tag(
                    repo_id=self.repo_id,
                    tag=new_version,
                    tag_message=f"Automated Model Training {new_version}",
                )
                
            if self.logger:
                self.logger.info(f"Successfully pushed artifacts to Hugging Face Hub as {new_version}.")
            return True
        except Exception as e:
            if self.logger:
                self.logger.exception(f"Error pushing to Hugging Face: {str(e)}")
            return False

    def list_models_versions(self):
        """Returns a list of versions (commits) available on Hugging Face for this repo."""
        if not self.token:
            return []
            
        try:
            # Check if repo exists first
            try:
                self.api.repo_info(repo_id=self.repo_id, repo_type="model")
            except:
                return []

            refs = self.api.list_repo_refs(repo_id=self.repo_id)
            # Filter and sort tags starting with 'v'
            tags = [tag for tag in refs.tags if tag.name.startswith("v")]
            tags = sorted(tags, key=lambda x: float(x.name.replace("v", "")), reverse=True)
            
            versions = []
            for idx, tag in enumerate(tags):
                versions.append({
                    "version": tag.name,         
                    "run_id": tag.commit,        # Tag points to a specific commit
                    "status": "Active" if idx == 0 else "Archive",
                    "date": "N/A",               # refs.tags doesn't expose date easily, can be enhanced if needed
                    "metrics": {}                
                })
                
            return versions
        except Exception as e:
            if self.logger:
                self.logger.exception(f"Error fetching commits from HF: {str(e)}")
            return []

    def download_model_version(self, commit_id):
        """Downloads a specific commit of the models from Hugging Face back into the local runtime."""
        from huggingface_hub import snapshot_download
        if not self.token:
            return False
            
        if self.logger:
            self.logger.info(f"Downloading model version {commit_id} from {self.repo_id}...")
            
        try:
            # Download the folder from the specific commit into apps/models locally
            snapshot_download(
                repo_id=self.repo_id,
                revision=commit_id,
                local_dir="apps",
                token=self.token,
                allow_patterns=["models/*"]
            )
            
            if self.logger:
                self.logger.info("Successfully downloaded models from Hugging Face Hub.")
            return True
        except Exception as e:
            if self.logger:
                self.logger.exception(f"Error downloading from Hugging Face: {str(e)}")
            return False

    def get_model_snapshot(self, tag_name=None):
        """
        Ensures a specific tag (or latest if None) is downloaded to the HF cache.
        Returns the path to the 'models' folder.
        """
        from huggingface_hub import snapshot_download
        if not self.token:
            return None
            
        try:
            # download_path will be something like ~/.cache/huggingface/hub/models--.../snapshots/...
            download_path = snapshot_download(
                repo_id=self.repo_id,
                revision=tag_name, #revision can be a tag name like 'v1.0'
                token=self.token,
                allow_patterns=["models/*"]
            )
            
            # The models are inside a 'models' subfolder in the repo
            models_path = os.path.join(download_path, "models")
            if os.path.exists(models_path):
                return models_path
            return download_path
        except Exception as e:
            if self.logger:
                self.logger.exception(f"Error resolving model snapshot: {str(e)}")
            return None
