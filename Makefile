.PHONY: all module run plot clear movie build_cuda build console visu

all: clear module

module: build_cuda
	CC=gcc && LDSHARED='$(shell python3 scripts/configure.py)' && python3 setup.py build
	python3 -m pip install .
# Clean up the temporary directory
	rm -rf cuda_build

run:
	mkdir -p wave_data
	python3 scripts/launch_sim.py

plot: 
	python3 scripts/plot_all.py

#rem is a command that does nothing in case the file doesnt exist. Windows...
clear:
	rm -rf wave_data wave_images wave.mp4 libmigration.lib *.obj temp.py build images cuda_build

	

movie:	
	ffmpeg -y -an -i wave_images/%5d.png -vcodec libx264 -pix_fmt yuv420p -profile:v baseline -level 3 -r 12 movies/wave.mp4

#build cuda library
build_cuda:
	mkdir -p cuda_build
	nvcc -rdc=true --compiler-options '-fPIC' -c -o cuda_build/temp.o src/simulate_kernel.cu
	nvcc -dlink --compiler-options '-fPIC' -o cuda_build/modeling.o cuda_build/temp.o -lcudart
	rm -f cuda_build/libmodeling.a
	ar cru cuda_build/libmodeling.a cuda_build/modeling.o cuda_build/temp.o
	ranlib cuda_build/libmodeling.a


NVCC_FLAGS = -O0 -I./src

build: src/argument_utils.c src/modeling_cmd.c src/simulate_kernel.cu src/getopt.c
	nvcc $(NVCC_FLAGS) $^ -o bin/modeling_cmd


console: 
	mkdir -p wave_data
	./bin/modeling_cmd -x 0.01 -y 0.01 -z 0.01 -X 100 -Y 100 -Z 100 -t 1e-8 -i 300 -s 1


visu: run plot movie

test: clear build console plot movie
