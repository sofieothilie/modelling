# export PATH := "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.43.34808/bin/Hostx64/x64":$(PATH)

all: clear module

module: build_cuda
	SET CC=cl && set LDSHARED='$(shell python scripts/configure.py)' && python setup.py build
	python -m pip install .
# Clean up the temporary directory
	@del /F /Q cuda_build\temp.obj cuda_build\modeling.obj cuda_build\libmodeling.lib || rem
	@rmdir /S /Q cuda_build 2>nul || rem

run:
	mkdir wave_data 2>nul || rem
	python scripts/launch_sim.py

plot: 
	python scripts/plot_all.py

#rem is a command that does nothing in case the file doesnt exist. Windows...
clear:
	rmdir /S /Q wave_data wave_images 2>nul || rem
	del /Q wave.mp4 2>nul || rem
	del /F /Q libmigration.lib *.obj temp.py 2>nul || rem
	rmdir /S /Q build 2>nul || rem
	rmdir /S /Q images 2>nul || rem
	del /F /Q cuda_build\temp.obj cuda_build\modeling.obj cuda_build\libmodeling.lib || rem
	rmdir /S /Q cuda_build 2>nul || rem
	exit /b 0	
	

movie:	
	ffmpeg -y -an -i wave_images/%5d.png -vcodec libx264 -pix_fmt yuv420p -profile:v baseline -level 3 -r 12 movies/wave.mp4


build_cuda:
	@mkdir cuda_build 2>nul || rem
	
# Compile the CUDA kernel
	nvcc --version
	nvcc -rdc=true -Xcompiler "/MD" -c -o cuda_build\temp.obj src/simulate_kernel.cu
	
# Link the object files
	nvcc -dlink -o cuda_build\modeling.obj cuda_build/temp.obj -Xlinker /NODEFAULTLIB:LIBCMT  -lcudart -lcudart_static
	
# Delete the previous libmodeling.lib if it exists
	@del /F /Q cuda_build\libmodeling.lib || rem
	
# Create the final library
	lib.exe /OUT:cuda_build/libmodeling.lib cuda_build/modeling.obj cuda_build/temp.obj 

build: src/argument_utils.c src/modeling_cmd.c src/simulate_kernel.cu
	


visu: run plot movie

