#!/bin/bash

echo "Setting up Car Damage Detection environment with Detectron2..."

# Create virtual environment
python3 -m venv car_damage_env

# Activate virtual environment
source car_damage_env/bin/activate

# Install requirements
pip install --upgrade pip
pip install -r requirements.txt

# Install Detectron2
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html

echo "Setup complete! Run 'source car_damage_env/bin/activate' to activate the environment." 