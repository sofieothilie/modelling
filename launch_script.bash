#!/bin/bash

# Simulation constants
SIMULATION_X=0.03
SIMULATION_Y=0.03
SIMULATION_Z=0.18

SENSOR_HEIGHT=0.023

PPW=8
ITERATIONS=4200
SNAPSHOT=5
PADDING=5

SENSOR_X=$(echo "3 - 0.385" | bc -l)

rm sensor_out/*

for i in $(seq 0 25); do
    SENSOR_Y=$(echo "scale=6; 0.065 + $i * 0.033" | bc -l)

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

    echo "$SENSOR_Y done."
done
