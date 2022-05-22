<details>
<summary>v3.1.1 (reference model)</summary>

___________________________________________________

- When to stop the training? 
  - train with all epoch and replace original model if Pearson correlation is better
- What is the original ensemble method? 
  - use 4 cross validation model to predict 4 score then average 4 score
- cv score in kaggle: xxx
</details>

<details>
<summary>Set up linux, NVIDIA and env in paperspace</summary>

_______________________________________
connect to server
```commandline
ssh paperspace@74.82.31.113 -i C:\Users\auyin11\key_pair\paperspace_key
```
Update and upgrade your Ubuntu instance
```commandline
sudo apt update
sudo apt upgrade 
```
Download and install NVIDIA driver
```commandline
wget https://us.download.nvidia.com/XFree86/Linux-x86_64/465.27/NVIDIA-Linux-x86_64-465.27.run
sudo bash NVIDIA-Linux-x86_64-465.27.run
```
Check gpu status
```commandline
nvidia-smi
```
Install cuda
```commandline
sudo apt install nvidia-cuda-toolkit (may need to reboost)
```
Download and install anaconda
```commandline
wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
bash Anaconda3-2022.05-Linux-x86_64.sh
```
Create env by txt file
```commandline
source anaconda3/bin/activate
conda create -n patent_comp --file linux_pantent_requirement.txt -c pytorch -c conda-forge
```
<br>
</details>
