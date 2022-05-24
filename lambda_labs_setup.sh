#!/bin/bash

echo "start to set up lambda labs env"

# Download and install anaconda
wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
yes | bash Anaconda3-2022.05-Linux-x86_64.sh -b

# create env by txt file
source anaconda3/bin/activate
yes | conda create -n patent_comp --file competition_patent_upload/linux_pantent_requirement.txt -c pytorch -c conda-forge

echo "you are ready to bash run_main.sh"