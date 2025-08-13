terraform {
  backend "local" {
    path = "../volume/terraform/terraform.tfstate"
  }
}
