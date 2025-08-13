#!/bin/bash
cd terraform
terraform destroy -lock=false
cd ..