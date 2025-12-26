# Docker env for Isaacsim 4.5.0
## Clone docker environment for issacsim
```
git clone --filter=blob:none --no-checkout https://github.com/tsuneka/isaacsim.git
cd isaacsim/
git sparse-checkout set docker_env
git checkout
ls # Ensure that a directory named docker_env exists.
```
You edit USERNAME in .env file. Name your username as you like.
## Docker build and run
Build a docker environment to run isaacsim. I created the docker environment by referring to the following cite:
<https://zenn.dev/eochair/articles/isaacsim_tutorial>

<https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/index.html>
```
cd docker_env/
sudo docker compose build gpu
sudo docker compose run -u <username> gpu
```
## Create virtual env for Issacsim
```
cd isaacsim/ 
# activate venv environment
source .venv/bin/activate 
# If not, install the following packages:
pip3 install requests
```
# Clone in your work spase
```
git clone https://github.com/tsuneka/isaacsim.git 
```
# Exec sample code
```
cd isaacsim/src/
# This is the multi-munipulator example.
python send_sample.py 
# This is the mobile robot with lidar.
python sample_simulate.py
```