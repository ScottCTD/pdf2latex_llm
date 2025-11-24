resource "google_artifact_registry_repository" "repo" {
  location      = var.region
  repository_id = var.repo_name
  description   = "Docker repository for pdf2latex images"
  format        = "DOCKER"

  depends_on = [google_project_service.artifact_registry]
}
