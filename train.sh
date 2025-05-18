#!/bin/bash

echo "Training Car Damage Detection Model..."

# Activate virtual environment
source car_damage_env/bin/activate

# Run training script
python train_car_damage_model.py

sleep 5

echo "Training completed!"

echo "evaluating model..."

python evaluate_car_damage_model.py

echo "evaluation completed!"