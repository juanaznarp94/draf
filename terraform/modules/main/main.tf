terraform {
  required_providers {
    null = {
      source  = "hashicorp/null"
      version = "3.2.3"
    }
  }
}

variable "runName" {}
variable "aws_access_key" {}
variable "aws_secret_key" {}
variable "aws_session_token" {}
variable "aws_region" {}
variable "bucket" {}
variable "type" {}
variable "key" {}
variable "security_group" {}
variable "priv_key" {}
variable "id" {}
variable "serverIP" {}
variable "num_partitions" {}
variable "num_server_rounds" {}

variable "fraction_fit"{}
variable "local_epochs"{}
variable "strategy" {}
variable "min_clients"{}
variable "strongness" {}

resource "aws_instance" "dlEc" {
  ami             = "ami-0c02fb55956c7d316"
  instance_type   = "t3.medium"
  key_name        = var.key
  security_groups = [var.security_group]
  root_block_device {
    volume_size = 32
  }
  provisioner "file" {
    source      = "../Flower"
    destination = "/home/ec2-user/Flower"

    connection {
      type        = "ssh"
      user        = "ec2-user"
      private_key = var.priv_key
      host        = self.public_ip
    }
  }

  provisioner "file" {
    destination="/home/ec2-user/Flower/.env"
    content = <<-EOT
    CLIENT=${var.id}
    NUM_SERVER_ROUNDS=${var.num_server_rounds}
    BUCKET=${var.bucket}
    AWS_SECRET_ACCESS_KEY=${var.aws_secret_key}
    AWS_SESSION_TOKEN=${var.aws_session_token}
    AWS_ACCESS_KEY_ID=${var.aws_access_key}
    AWS_REGION=${var.aws_region}
    MIN_CLIENTS=${var.min_clients}
    FRACTION_FIT=${var.fraction_fit}
    LOCAL_EPOCHS=${var.local_epochs}
    SERVER_IP=${var.serverIP}
    RUN_NAME=${var.runName}
    TYPE=client
    STRATEGY=${var.strategy}
    STRONGNESS=${var.strongness}
  EOT

    connection {
      type        = "ssh"
      user        = "ec2-user"
      private_key = var.priv_key
      host        = self.public_ip
    }
  }

  provisioner "remote-exec" {
    inline = [
      "sudo yum update -y",
      "sudo yum install -y docker",
      "sudo usermod -a -G docker ec2-user",
      "id ec2-user",
      #"newgrp docker",
      "sudo systemctl enable docker.service",
      "sudo systemctl start docker.service",
      "sudo curl -SL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 -o /usr/libexec/docker/cli-plugins/docker-compose",
      "sudo chmod +x /usr/libexec/docker/cli-plugins/docker-compose",
      "docker compose version",
      "cd /home/ec2-user/Flower && sudo docker compose up -d",
    ]
    connection {
      type        = "ssh"
      user        = "ec2-user"
      private_key = var.priv_key
      host        = self.public_ip
    }
  }
}




resource "null_resource" "ssh_script_trigger" {
  triggers = {
    ip        = aws_instance.dlEc.public_ip
    timestamp = timestamp()
  }
}

resource "local_file" "ssh_script" {
  depends_on = [null_resource.ssh_script_trigger]

  content  = "ssh -i ~/.ssh/dl2Ec2Key ec2-user@${aws_instance.dlEc.public_ip}"
  filename = "${path.module}/../../../ssh-scripts/${var.type}Id_${var.id}.sh"
}


resource "null_resource" "make_script_executable" {
  depends_on = [local_file.ssh_script]

  provisioner "local-exec" {
    command = "chmod +x ${local_file.ssh_script.filename}"
  }
}

