provider "heroku" {
  email   = var.heroku_email
  api_key = var.heroku_api_key
}

resource "heroku_app" "app" {
  name   = var.app_name
  region = "us"
  stack  = "heroku-22"
}

resource "heroku_build" "build" {
  app_id     = heroku_app.app.id
  buildpacks = filelist("${path.module}/config.buildpacks")

  source {
    url     = var.source_url
    version = "v1"
  }
}

output "app_url" {
  value = "https://${heroku_app.app.name}.herokuapp.com"
}

