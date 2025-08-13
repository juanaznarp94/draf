terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
    local = {
      source  = "hashicorp/local"
      version = "2.5.2"
    }
  }
}

provider "aws" {
  region = var.aws_region
  access_key = var.aws_access_key
  secret_key = var.aws_secret_key
  token = var.aws_session_token
}
resource "tls_private_key" "deployer" {
  algorithm = "RSA"
  rsa_bits  = 4096
}
resource "aws_key_pair" "deployer" {
  key_name   = "dl_project_key"
  public_key =tls_private_key.deployer.public_key_openssh
}




module "serverVm" {
  source = "./modules/server"

  id             = 0
  key            = aws_key_pair.deployer.key_name
  priv_key       = tls_private_key.deployer.private_key_openssh
  security_group = aws_security_group.ec2_sg.name
  type           = "server"
  num_partitions = var.clients+1
  num_server_rounds=var.num_server_rounds
  fraction_fit=var.fraction_fit
  local_epochs=var.local_epochs
  strategy=var.strategy
  min_clients=var.clients
  runName = var.runName
  bucket = aws_s3_bucket.bucket.id
  aws_access_key = var.aws_access_key
  aws_region = var.aws_region
  aws_secret_key = var.aws_secret_key
  aws_session_token = var.aws_session_token
  strongness = var.strongness
}


resource "aws_s3_bucket" "bucket" {
  bucket = "distbuckettrainingdatamarc"
  force_destroy = true
}

module "clientVms" {
  count  = var.clients
  source = "./modules/main"

  id             = count.index+1
  serverIP       = module.serverVm.serverIP
  key            = aws_key_pair.deployer.key_name
  priv_key       = tls_private_key.deployer.private_key_openssh
  security_group = aws_security_group.ec2_sg.name
  type           = "client"
  bucket = aws_s3_bucket.bucket.id

  num_partitions     = var.clients+1
  num_server_rounds  = var.num_server_rounds
  fraction_fit       = var.fraction_fit
  local_epochs       = var.local_epochs
  strategy           = var.strategy
  min_clients        = var.clients
  runName = var.runName
  aws_access_key = var.aws_access_key
  aws_region = var.aws_region
  aws_secret_key = var.aws_secret_key
  aws_session_token = var.aws_session_token

  depends_on = [aws_key_pair.deployer]
  strongness = var.strongness
}


resource "aws_security_group" "ec2_sg" {
  name        = "dlProjectsg"
  description = "Allow SSH and outbound internet"
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  ingress {
    from_port = 9090
    to_port = 9100
    protocol = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}



output "private_key_pem" {
  value     = tls_private_key.deployer.private_key_pem
  sensitive = true
}
resource "local_file" "private_key_pem" {
  filename        = pathexpand("~/.ssh/dl2Ec2Key")
  content         = tls_private_key.deployer.private_key_pem
  file_permission = "0600"
}







