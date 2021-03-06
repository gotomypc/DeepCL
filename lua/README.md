# Lua Wrappers

## Concept

Lua wrappers are available.  [luajit](http://luajit.org) is becoming big in machine-learning.

## Demo

* For a demo of high-level functions, to create a network and train it, you can have a look at the method `test_basic` in [test_deepcl.lua](test_deepcl.lua)
* For a demo of constructing layers by hand, and handling low-level propagating, batch by batch, you can look at the method `test_lowlevel`, in the same module, ie in [test_deepcl.lua](test_deepcl.lua).
* For a demo of q-learning, you can look at [test_qlearning.lua](test_qlearning.lua)

## Installation from luarocks

* There is a source rock available on luarocks [luadeepcl](http://luarocks.org/modules/hughperkins/luadeepcl):

```
luarocks install --server=http://luarocks.org luadeepcl
```
* This builds from source, just as for the below, so this does have the same pre-requisites as building 
directly from github source, ie:
  * cmake
  * lua development libraries (eg `sudo apt-get install liblua5.1-0-dev`)
  * a C++ compiler, supporting c++0x
* You'll also need a working OpenCL-enabled platform, eg OpenCL-enabled GPU, or OpenCL-enabled CPU

## To build, linux

### Pre-requisites

* cmake
* g++, supporting c++0x, on linux
* lua development libraries (eg `sudo apt-get install liblua5.1-0-dev`)

### Procedure

From this directory:
```bash
mkdir build
cd build
cmake ..
make -j 4
```

## To build, Windows (untested)

### Pre-requisites

* cmake
* visual studio 2010, or later
* lua development libraries

### Procedure

- open cmake, point at the `lua` directory, and set to build in the `lua\build` subdirectory
  - accept `yes` to create the new directory
  - click `configure`
  - select appropriate generator, eg Visual Studio 2010, according to which one you have
  - click `generate`
- open visual studio, and load any of the projects in the `build` directory
  - change release type to `Release`
  - choose `build` from the `build` menu

## To run

### Pre-requisites

* have done build, or downloaded binaries
* An OpenCL-compatible driver installed, and OpenCL-compatible GPU

### On linux

From this directory, the one with this README.md in, do eg:
```bash
LUA_CPATH=build/?.so luajit test_lua.lua
```
or:
```bash
LUA_CPATH=build/?.so luajit test_qlearning.lua
```

### On Windows

Something like (not tested, at all...), from this directory, the one with this README.md in:
```cmd
set LUA_CPATH=build/win32/Release/?.dll
luajit test_lua.lua
```

## Unit-testing

* The source-code includes the thirdparty lua unit-test tool luaunit.  [test_deepcl.lua](test_deepcl.lua)
creates some first initial unit tests

## To build a rock

To build a source rock, use linux:
* first set the version in version.txt to your desired version
* then run:
```
./pack.sh
```
* whilst `pack.sh` only runs on linux, hopefully the resulting rock should be cross-platform.  Hopefully.  Let
me know any issues please :-)

## Development

* If you want to update the wrappers, you should install [swig](http://www.swig.org), and turn on the option 'RUN_SWIG' in cmake

