

all: module test


module:
	python3 setup.py build
	sudo python3 setup.py install

test:
	python3 launch_sim.py


plot: 
	python3 plot_all.py

visu: clear module test 
	make plot
	ffmpeg -y -an -i images/%5d.png -vcodec libx264 -pix_fmt yuv420p -profile:v baseline -level 3 -r 12 wave.mp4

clear:
	-rm -fr data/* images/ wave.mp4

movie:	
	ffmpeg -y -an -i images/%5d.png -vcodec libx264 -pix_fmt yuv420p -profile:v baseline -level 3 -r 12 wave.mp4
