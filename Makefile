.PHONY: all run plot clear movie build debug save

all: test

SIMULATION_X = 0.02
SIMULATION_Y = 0.5
SIMULATION_Z = 0.224#0.224

SENSOR_X = 1.40223
SENSOR_Y = 0.885
# SENSOR_X = 2.615 #still needs to add some shift to be at the center of the receiver, like 1cm
# SENSOR_Y = 0.494 #here also
SENSOR_HEIGHT = 0.023

PPW = 6
ITERATIONS = 1#unused, will be overwritten by the correct round trip time
SNAPSHOT = 5
PADDING = 5
RTM = 1

plot: 
	@echo "Started plotting..."
	@python3 scripts/plot_all.py $(PPW)  $(SNAPSHOT) $(SIMULATION_X) $(SIMULATION_Y) $(SIMULATION_Z)

clear:
	@rm -rf wave_data wave_images wave.mp4 libmigration.lib *.obj temp.py build images cuda_build

movie:	
	@echo -n "Started generating movie... "
	@mkdir -p movies
	@ffmpeg -y -an -i wave_images/%5d.png -vcodec libx264 -pix_fmt yuv420p -profile:v baseline -level 3 -r 12 movies/wave_$(shell date +%Y%m%d_%H%M%S).mp4 > /dev/null 2>&1
	@echo "Done."



NVCC_FLAGS =  -O3 -Wno-deprecated-gpu-targets   -I./src

build: src/argument_utils.c src/model_cli.c src/getopt.c src/memory_management.cu src/utils.cu src/simulation.cu
	@echo -n "Compiling... "
	@nvcc $(NVCC_FLAGS) $^ -o  bin/model_cli >  /dev/null 
	@echo "Done."

debug: src/argument_utils.c src/model_cli.c src/getopt.c src/memory_management.cu src/utils.cu src/simulation.cu
	@nvcc $(NVCC_FLAGS) $^ -g -G -o  bin/model_cli >  /dev/null 
	@echo "Debug Compilation Successful\n"

run: 
	@mkdir -p wave_data
	@mkdir -p sensor_out
	./bin/model_cli -x $(SIMULATION_X) -y $(SIMULATION_Y) -z $(SIMULATION_Z) -X $(SENSOR_X) -Y $(SENSOR_Y) -Z $(SENSOR_HEIGHT) -p $(PPW) -i $(ITERATIONS) -s $(SNAPSHOT) --padding $(PADDING) -R $(RTM)

#only print launch info, without running it
info: 
	@./bin/model_cli --print-info -x $(SIMULATION_X) -y $(SIMULATION_Y) -z $(SIMULATION_Z) -X $(SENSOR_X) -Y $(SENSOR_Y) -Z $(SENSOR_HEIGHT) -p $(PPW) -i $(ITERATIONS) -s $(SNAPSHOT) --padding $(PADDING)

save:
	@if [ -z "$(path)" ]; then \
		echo "Error: filename parameter is required. Use 'make save path=<name>'"; \
		exit 1; \
	fi
	@echo -n "Saving data... "
	@mkdir -p $(path)
	@mv wave_data $(path)/wavefield
	@mv sensor_out $(path)/sensor_out
	@echo "Done."


test: clear build run plot movie

