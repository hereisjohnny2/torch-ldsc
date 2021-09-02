# Pytorch - LIBLDSC Integration

## About

This project is an attempt to integrate both **pytorch** and **libldsc** libraries to create a rock image segmentation model. The **libldsc** is a library developed by students and professors in the [Laboratório de Desenvolvimento de Software Científico](https://github.com/ldsc) at the [Universidade Estadual do Norte Fluminense Darcy Ribeiro](https://uenf.br/). This library has a set of objects related to porous media images analisys, hear transfer, mathematical, statistical and numeric problems solution.

## Technologies

 - [C++](https://cplusplus.com/)
 - [CMake](https://cmake.org/)
 - [LibTorch](https://pytorch.org/)
 - [LibLDSC](https://github.com/ldsc/lib_ldsc)

## Running the project

The pytorch library should be installed locally to compile and run the project. Follow the instruction the pytorch C++ [documentation](https://pytorch.org/tutorials/advanced/cpp_frontend.html). Then change the files location in the `CMakeLists.txt` file at the project root.

```cmake
list(APPEND CMAKE_PREFIX_PATH "/your/torch/location/libtorch")
```

This project also demands CMake building system to compile the source code. So after clone this repository create a "build" folder in the project root:

```bash
$ mkdir build
```

Then run the cmake build command to create the file needed:

```bash
$ cmake -B build -S .
```

Next, go inside the build folder and run the Makefile created by CMake:

```bash
$ cd build
$ make
```

Finally, run the executable named `torch-ldsc`.

```bash
$ ./torch-ldsc
```

So far, you can run the executable with the `learning rate` and the `batch size` as parameters. This can make easier the training process:

```bash
$ ./torch-ldsc 0.04 5
```

## Tests
