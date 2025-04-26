.PHONY: all run plot clear movie  build

all: test



plot: 
	python3 scripts/plot_all.py

clear:
	rm -rf wave_data wave_images wave.mp4 libmigration.lib *.obj temp.py build images cuda_build

	

movie:	
	ffmpeg -y -an -i wave_images/%5d.png -vcodec libx264 -pix_fmt yuv420p -profile:v baseline -level 3 -r 12 movies/wave.mp4



NVCC_FLAGS =-O3 -Wno-deprecated-gpu-targets   -I./src

build: src/argument_utils.c src/model_cli.c src/getopt.c src/memory_management.cu src/utils.cu src/simulation.cu
	nvcc $(NVCC_FLAGS) $^ -o  bin/model_cli
	@echo "Compilation Successful"

debug: src/argument_utils.c src/model_cli.c src/getopt.c src/memory_management.cu src/utils.cu src/simulation.cu
	nvcc $(NVCC_FLAGS) $^ -g -G -lineinfo -o  bin/model_cli
	@echo "Debug Compilation Successful"

run: 
	@mkdir -p wave_data
	./bin/model_cli -x 0.1 -y 0.1 -z 0.4 -p 5 -t 8.5e-8 -i 10 -s 1 --padding 5


test: clear build run plot movie

