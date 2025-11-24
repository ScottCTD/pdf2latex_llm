variable "project_id" {
  description = "The ID of the project in which to provision resources."
  type        = string
}

variable "region" {
  description = "The region in which to provision resources."
  type        = string
  default     = "us-central1"
}


variable "repo_name" {
  description = "The name of the Artifact Registry repository to create."
  type        = string
  default     = "pdf2latex-repo"
}
