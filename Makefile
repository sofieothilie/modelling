.PHONY: all run plot clear movie  build

all: test

SIMULATION_X = 0.01
SIMULATION_Y = 0.01
SIMULATION_Z = 0.07

SENSOR_X = -1.0
SENSOR_Y = -1.0
SENSOR_HEIGHT = 0.1

PPW = 6
ITERATIONS = 4000
SNAPSHOT = 3
PADDING = 5

plot: 
	@echo "Started plotting..."
	@python3 scripts/plot_all.py $(PPW)  $(SNAPSHOT) $(PADDING)

clear:
	@rm -rf wave_data wave_images wave.mp4 libmigration.lib *.obj temp.py build images cuda_build

	

movie:	
	@echo -n "Started generating movie... "
	@ffmpeg -y -an -i wave_images/%5d.png -vcodec libx264 -pix_fmt yuv420p -profile:v baseline -level 3 -r 12 movies/wave.mp4 > /dev/null  2>&1
	@echo "Done."



NVCC_FLAGS =  -O3 -Wno-deprecated-gpu-targets   -I./src

build: src/argument_utils.c src/model_cli.c src/getopt.c src/memory_management.cu src/utils.cu src/simulation.cu
	@echo -n "Compiling... "
	@nvcc $(NVCC_FLAGS) $^ -o  bin/model_cli >  /dev/null 
	@echo "Done."

debug: src/argument_utils.c src/model_cli.c src/getopt.c src/memory_management.cu src/utils.cu src/simulation.cu
	@nvcc $(NVCC_FLAGS) $^ -g -G -lineinfo -o  bin/model_cli
	@echo "Debug Compilation Successful\n"

run: 
	@mkdir -p wave_data
	@./bin/model_cli -x $(SIMULATION_X) -y $(SIMULATION_Y) -z $(SIMULATION_Z) -X $(SENSOR_X) -Y $(SENSOR_Y) -Z $(SENSOR_HEIGHT) -p $(PPW) -i $(ITERATIONS) -s $(SNAPSHOT) --padding $(PADDING)

info: 
	@./bin/model_cli --print-info -x $(SIMULATION_X) -y $(SIMULATION_Y) -z $(SIMULATION_Z) -X $(SENSOR_X) -Y $(SENSOR_Y) -Z $(SENSOR_HEIGHT) -p $(PPW) -i $(ITERATIONS) -s $(SNAPSHOT) --padding $(PADDING)



test: clear build run plot movie

