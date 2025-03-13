

export PATH := "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.43.34808/bin/Hostx64/x64":$(PATH)

all: module

module: build_cuda
	SET CC=cl && set LDSHARED='$(shell python scripts/configure.py)' && python setup.py build
	python -m pip install .

run:
	mkdir data 2>nul || echo ""
	python launch_sim.py


plot: 
	python plot_all.py

visu: clear module test 
	make plot
	ffmpeg -y -an -i images/%5d.png -vcodec libx264 -pix_fmt yuv420p -profile:v baseline -level 3 -r 12 wave.mp4

clear:
	rmdir /S /Q data images 2>nul || echo ""
	del /Q wave.mp4 2>nul || echo ""
	del /F /Q libmigration.lib *.obj temp.py 2>nul || echo Files not found
	rmdir /S /Q build 2>nul || echo Directory not found
	rmdir /S /Q images 2>nul || echo Directory not found
	exit /b 0	

movie:	
	ffmpeg -y -an -i images/%5d.png -vcodec libx264 -pix_fmt yuv420p -profile:v baseline -level 3 -r 12 wave.mp4

build_cuda:
	nvcc --version
	nvcc -rdc=true -Xcompiler "/MD" -c -o temp.obj simulate_kernel.cu
	nvcc -dlink -o modeling.obj temp.obj -Xlinker /NODEFAULTLIB:LIBCMT -L"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/lib/x64" -lcudart -lcudart_static
	del /F /Q libmodeling.lib
	C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.43.34808/bin/Hostx64/x64/lib.exe /OUT:libmodeling.lib modeling.obj temp.obj 

test: run plot

visu: run plot movie