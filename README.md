# Skylarklabs Cab violence detection

## Download Dataset
```bash

!pip install gdown > /dev/null
!apt-get install p7zip-full -y > /dev/null

# https://drive.google.com/file/d/14FficMOD36m_IuOfpQg05qaiTjMNuKZG/view?usp=sharing
!gdown --id 14FficMOD36m_IuOfpQg05qaiTjMNuKZG
!unzip -o CAR_VIOLENCE_DATASET_final.zip > /dev/null
```

## Clone the Repo
```bash
!git clone https://github.com/Jovian-Dsouza/Skylarklabs-cab_violence_detection
%cd Skylarklabs-cab_violence_detection
```

## Start Training
```bash
!python autotrain.py
```