#!/bin/bash

# Simulation constants
SIMULATION_X=0.01
SIMULATION_Y=0.05
SIMULATION_Z=0.224

SENSOR_HEIGHT=0.023

PPW=8
ITERATIONS=1#unused
SNAPSHOT=5
PADDING=5

SENSOR_X=$(echo "3 - 0.385" | bc -l)

rm sensor_out/*

for i in $(seq 0 25); do
    SENSOR_Y=$(echo "scale=6; 0.065 + $i * 0.033" | bc -l)
    
    mkdir -p wave_data
    mkdir -p sensor_out
    ./bin/model_cli \
        -x "$SIMULATION_X" \
        -y "$SIMULATION_Y" \
        -z "$SIMULATION_Z" \
        -X "$SENSOR_X" \
        -Y "$SENSOR_Y" \
        -Z "$SENSOR_HEIGHT" \
        -p "$PPW" \
        -i "$ITERATIONS" \
        -s "$SNAPSHOT" \
        --padding "$PADDING" \

    make save path="simulation_z0_ppw8_x.005_y.02/sensor_${SENSOR_Y}"
    echo "$SENSOR_Y done."
done
