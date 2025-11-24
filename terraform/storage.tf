resource "random_id" "bucket_suffix" {
  byte_length = 4
}

resource "google_storage_bucket" "dataset_bucket" {
  name          = "pdf2latex-dataset-${random_id.bucket_suffix.hex}"
  location      = var.region
  force_destroy = false

  uniform_bucket_level_access = true
}

output "bucket_name" {
  value = google_storage_bucket.dataset_bucket.name
}

