== Memo on compiling `heat_2d.cu` on Ubuntu Linux 20.04 ==

(1) Install CUDA toolkit.

```
sudo apt install nvidia-cuda-toolkit
```

This will install CUDA toolkit version 10.2 .

(2) Install GLUT dev package.

```
sudo apt-get install freeglut3-dev
```

(3) Add extra compile options.

nvcc -o heat_2d.ex heat_2d.cu -lGL -lglut


BTW: In order to run heat_2d.ex on  another target Ubuntu 20.04 machine, 
the target machine needs to have `libglut.so.3`. This can be achieved by

```
sudo apt-get install freeglut3
```
