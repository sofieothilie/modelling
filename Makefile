.PHONY: all run plot clear movie  build

all: test



plot: 
	python3 scripts/plot_all.py

clear:
	rm -rf wave_data wave_images wave.mp4 libmigration.lib *.obj temp.py build images cuda_build

	

movie:	
	ffmpeg -y -an -i wave_images/%5d.png -vcodec libx264 -pix_fmt yuv420p -profile:v baseline -level 3 -r 12 movies/wave.mp4



NVCC_FLAGS =-O3 -Wno-deprecated-gpu-targets   -I./src

build: src/argument_utils.c src/modeling_cmd.c src/getopt.c src/memory_management.cu src/utils.cu src/simulation.cu
	nvcc $(NVCC_FLAGS) $^ -o  bin/modeling_cmd
	@echo "Compilation Successful"

debug: src/argument_utils.c src/modeling_cmd.c src/getopt.c src/memory_management.cu src/utils.cu src/simulation.cu
	nvcc $(NVCC_FLAGS) $^ -g -G -lineinfo -o  bin/modeling_cmd
	@echo "Debug Compilation Successful"

console: 
	@mkdir -p wave_data
	./bin/modeling_cmd -x 0.02 -y 0.02 -z 0.02 -h 0.0002 -t 2e-8 -i 1000 -s 5


test: clear build console plot movie

