# Basic Image Restoration application
## Abstract
The image restoration application implements and analyzes few image restoration techniques, without using existing image processing libraries. The application provides an easy to use GUI built on PyQt5 to perform these operations on grayscale and colour images, and is completely implemented in Python. It was tested against a set of degraded images with varying levels of noise (AWGN) and known/ unknown blur kernels.   

The image restoration techniques implemented are:
* Inverse filtering
* Truncated inverse filtering
* Minimum mean square error (Weiner) filtering
* Constrained least squares filtering
## Dependencies
- python v3
- PyQt5
- python libraries : opencv (to read/ save images), numpy, matplotlib
## Instructions to run
~~~~
python3 main.py
~~~~
Sample input images are available in images folder.  

## Results
A screenshot of the application is given below.

![Basic Image Restoration Application Screenshot]()

A demo video of the application is available [here]().  
A detailed report of the application is available [here]().

## References
[1] Gonzalez, Rafael C., and Woods, Richard E. "Digital image processing. 3E" (2008).  
[2] http://webdav.is.mpg.de/pixel/benchmark4camerashake/  
[3] https://elementztechblog.wordpress.com/2015/04/14/getting-started-with-pycharm-and-qt-4-designer/  
[4] https://docs.scipy.org/doc/numpy/reference/  
[5] http://noise.imageonline.co/  
[6] Hore, Alain, and Djemel Ziou. "Image quality metrics: PSNR vs. SSIM." Pattern recognition (icpr), 2010 20th international conference on. IEEE, 2010.  
[7] https://github.com/shyama95/image-restoration
