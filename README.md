# Direct Sparse Odometry with Stereo Cameras on ARM devices

1.Direct Sparse Odometry

	Direct Sparse Odometry is crafted by J. Engel, V. Koltun, D. Cremers.
	See https://vision.in.tum.de/dso for more information.

2.Direct Sparse Odometry with Stereo Cameras

	The stereo version of Direct Sparse Odometry is developed by Jiatian WU, Degang YANG, Qinrui YAN, Shixin LI.
	See https://github.com/HorizonAD/stereo_dso for more information.

3.Features

	The original stereo dso does not support ARM devices(with sse2neon removed). I added this package back and it works well on my Jetson TX1.
	Some modification was also done to the example for test purpose.

4.Usage 

	Open a terminal under the project directory,
	bash INSTALL
	Alter the yaml file,
	bash run.sh
	
