#!/bin/bash

# Example 1: Testing with a clean image (Make sure you have this image in a 'samples' folder)
echo "Testing Clean Image..."
curl -X POST -F "image=@../samples/maize_rust.jpg" http://localhost:8000/predict

# Example 2: Testing the Robustness Bonus with a noisy field shot
echo -e "\n\nTesting Noisy Field Image..."
curl -X POST -F "image=@../samples/noisy_field_shot.jpg" http://localhost:8000/predict