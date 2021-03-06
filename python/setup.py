# Copyright Hugh Perkins 2015 hughperkins at gmail
#
# This Source Code Form is subject to the terms of the Mozilla Public License, 
# v. 2.0. If a copy of the MPL was not distributed with this file, You can 
# obtain one at http://mozilla.org/MPL/2.0/.

#from distutils.core import setup
import os
import os.path
import sysconfig
import sys
import glob
import platform
from setuptools import setup
#from distutils.extension import Extension
from setuptools import Extension
import distutils.dir_util
import distutils.file_util

cython_present = False
try:
    from Cython.Build import cythonize
    cython_present = True
except ImportError:
    pass

pypandoc_present = False
try:
    import pypandoc
    pypandoc_present = True
except ImportError:
    pass

#print ( sys.argv )
# if any of sys.argv is bdist or sdist or bdist_egg, then lets copy everything to
# a subfolder of us called 'mysrc', since '..' paths dont work well..
# otherwise, let's just assume this folder already contains our source :-)
docopy = False
for arg in sys.argv:
    if arg in ('sdist','bdist','bdist_egg','build_ext'):
        docopy = True

if docopy:
    if not os.path.isdir('mysrc'):
        os.makedirs('mysrc')
    for thisdir in ['../src','../OpenCLHelper','../qlearning',
            '../OpenCLHelper/thirdparty/clew/src']: # copy everything..
        for thisfile in os.listdir(thisdir):
            #print(thisfile)
            thisfilepath = thisdir +'/' + thisfile
            if os.path.isfile(thisfilepath):
                distutils.file_util.copy_file( thisfilepath, 'mysrc/' + thisfile )
    distutils.file_util.copy_file( '../jenkins/version.txt', 'version.txt' )

#        distutils.dir_util.copy_tree( thisdir, 'mysrc' )

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

def get_so_suffix():
    if sysconfig.get_config_var('SOABI') != None:
        return "." + sysconfig.get_config_var('SOABI')
    return ""

if pypandoc_present:
    pypandoc.convert('README.md', 'rst', outputfile = 'README.rst' )

def my_cythonize(extensions, **_ignore):
    #newextensions = []
    for extension in extensions:
        print(extension.sources)
        should_cythonize = False
        sources = []
        for sfile in extension.sources:
            path, ext = os.path.splitext(sfile)
            if ext in ('.pyx', '.py'):
                should_cythonize = True
                if not cython_present:
                   # if extension.language == 'c++':
                        ext = '.cpp'
                    #else:
                     #   ext = '.c'
            if sfile == 'PyDeepCL.c':
                ext = '.cpp' # hack for now... not sure how to fix this cleanly
                             # yet
            sfile = path + ext
            if sfile.startswith('..'):
                # use mysrc instead
                basename = os.path.basename(sfile)
                sfile = 'mysrc/' + basename
            sources.append(sfile)
        #print(should_cythonize)
        if should_cythonize and cython_present:
            print('cythonizing...')
            cythonize(extension)
        extension.sources[:] = sources    
        #newextensions.append( extension )
    return extensions

def no_cythonize(extensions, **_ignore):
    for extension in extensions:
        sources = []
        for sfile in extension.sources:
            path, ext = os.path.splitext(sfile)
            #print('path,ext',path,ext)
            if ext in ('.pyx', '.py'):
                #if extension.language == 'c++':
                    ext = '.cpp'
                #else:
                #    ext = '.c'
            if sfile == 'PyDeepCL.c':
                ext = '.cpp' # hack for now... not sure how to fix this cleanly
                             # yet
            sfile = path + ext
            if sfile.startswith('..'):
                # use mysrc instead
                basename = os.path.basename(sfile)
                sfile = 'mysrc/' + basename
            sources.append(sfile)
            print('appending source ', sfile )
        extension.sources[:] = sources    
    return extensions

# from http://stackoverflow.com/questions/14320220/testing-python-c-libraries-get-build-path
def distutils_dir_name(dname):
    """Returns the name of a distutils build directory"""
    f = "{dirname}.{platform}-{version[0]}.{version[1]}"
    return f.format(dirname=dname,
                    platform=sysconfig.get_platform(),
                    version=sys.version_info)
 
def lib_build_dir():
    return os.path.join('build', distutils_dir_name('lib'))

deepcl_sourcestring = """LayerMaker.cpp NeuralNetMould.cpp
     ConvolutionalLayer.cpp NeuralNet.cpp Layer.cpp InputLayer.cpp
    Propagate1.cpp Propagate.cpp Propagate2.cpp Propagate3.cpp LayerDimensions.cpp
    Propagate4.cpp ActivationFunction.cpp SquareLossLayer.cpp LossLayer.cpp BackpropWeights2.cpp
    BackpropWeights2Cpu.cpp BackpropErrorsv2.cpp BackpropErrorsv2Cpu.cpp
    BackpropWeights2Naive.cpp BackpropErrorsv2Naive.cpp BackpropWeights2Scratch.cpp
    CrossEntropyLoss.cpp SoftMaxLayer.cpp FullyConnectedLayer.cpp  EpochMaker.cpp
    PoolingPropagate.cpp PoolingPropagateCpu.cpp PoolingLayer.cpp PoolingBackprop.cpp
    PoolingBackpropCpu.cpp PoolingPropagateGpuNaive.cpp BackpropWeights2ScratchLarge.cpp
    BatchLearner.cpp NetdefToNet.cpp NetLearner.cpp stringhelper.cpp NormalizationLayer.cpp
    RandomPatches.cpp RandomTranslations.cpp NorbLoader.cpp MultiNet.cpp
    Trainable.cpp InputLayerMaker.cpp ConvolutionalMaker.cpp RandomTranslationsMaker.cpp
    RandomPatchesMaker.cpp NormalizationLayerMaker.cpp FullyConnectedMaker.cpp
    PoolingMaker.cpp PatchExtractor.cpp Translator.cpp GenericLoader.cpp Kgsv2Loader.cpp
    BatchLearnerOnDemand.cpp NetLearnerOnDemand.cpp BatchProcess.cpp WeightsPersister.cpp
    PropagateFc.cpp BackpropErrorsv2Cached.cpp PropagateByInputPlane.cpp
    PropagateExperimental.cpp PropagateAuto.cpp PropagateCpu.cpp Propagate3_unfactorized.cpp
    PoolingBackpropGpuNaive.cpp ForceBackpropLayerMaker.cpp ForceBackpropLayer.cpp
    MnistLoader.cpp OnDemandBatcher.cpp Batcher.cpp NetAction.cpp ActivationLayer.cpp
    ActivationMaker.cpp
    ActivationPropagate.cpp ActivationPropagateCpu.cpp ActivationPropagateGpuNaive.cpp
    ActivationBackprop.cpp ActivationBackpropCpu.cpp ActivationBackpropGpuNaive.cpp
    DropoutMaker.cpp DropoutLayer.cpp DropoutPropagate.cpp DropoutBackprop.cpp
    DropoutPropagateCpu.cpp DropoutPropagateGpuNaive.cpp
    DropoutBackpropCpu.cpp DropoutBackpropGpuNaive.cpp
    CopyBuffer.cpp MultiplyBuffer.cpp
""" 
deepcl_sources_all = deepcl_sourcestring.split()
deepcl_sources = []
for source in deepcl_sources_all:
    deepcl_sources.append(source)

openclhelpersources = list(map( lambda name : 'mysrc/' + os.path.basename( name ), [ 'OpenCLHelper/OpenCLHelper.cpp',
        'OpenCLHelper/deviceinfo_helper.cpp', 'OpenCLHelper/platforminfo_helper.cpp',
        'OpenCLHelper/CLKernel.cpp', 'OpenCLHelper/thirdparty/clew/src/clew.c' ] ))
print(openclhelpersources)
print(isinstance( openclhelpersources, list) )

compile_options = []
osfamily = platform.uname()[0]
if osfamily == 'Windows':
   compile_options.append('/EHsc')
elif osfamily == 'Linux':
   compile_options.append('-std=c++0x')
   compile_options.append('-g')
else:
   pass
   # put other options etc here if necessary

runtime_library_dirs = []
libraries = []
if osfamily == 'Linux':
    runtime_library_dirs= ['.']

if osfamily == 'Windows':
    libraries = ['winmm']

if cython_present:
    my_cythonize = cythonize
else:
    my_cythonize = no_cythonize

#libraries = [
#    ("OpenCLHelper", {
#        'sources': openclhelpersources + ['dummy_openclhelper.cpp'],
#        'include_dirs': ['DeepCL/OpenCLHelper'],
#        'extra_compile_args': compile_options,
##        define_macros = [('OpenCLHelper_EXPORTS',1)],
##        libraries = []
##        language='c++'
#        }
#    )
#]

ext_modules = [
#    Extension("_OpenCLHelper",
#        sources = openclhelpersources + ['dummy_openclhelper.cpp'],
#        include_dirs = ['DeepCL/OpenCLHelper'],
#        extra_compile_args=compile_options,
#        define_macros = [('OpenCLHelper_EXPORTS',1),('MS_WIN32',1)],
##        libraries = []
##        language='c++'
#    )
#    Extension("libDeepCL",
#        list(map( lambda name : 'DeepCL/src/' + name, deepcl_sources)), # +
##            glob.glob('DeepCL/src/*.h'),
#        include_dirs = ['DeepCL/src','DeepCL/OpenCLHelper'],
#        extra_compile_args = compile_options,
#        library_dirs = [ lib_build_dir() ],
#        libraries = [ "OpenCLHelper" + get_so_suffix() ],
#        define_macros = [('DeepCL_EXPORTS',1)],
#        runtime_library_dirs=runtime_library_dirs
##        language='c++'
#    ),
    Extension("PyDeepCL",
              sources=["PyDeepCL.pyx", 'CyWrappers.cpp'] 
                + openclhelpersources
                + list(map( lambda name : 'mysrc/' + name, deepcl_sources))
                + ['mysrc/QLearner.cpp','mysrc/array_helper.cpp'], 
#                glob.glob('DeepCL/OpenCLHelper/*.h'),
              include_dirs = ['mysrc'],
              libraries= libraries,
              extra_compile_args=compile_options,
        define_macros = [('DeepCL_EXPORTS',1),('OpenCLHelper_EXPORTS',1)],
#              extra_objects=['cDeepCL.pxd'],
#              library_dirs = [lib_build_dir()],
              runtime_library_dirs=runtime_library_dirs,
              language="c++"
    )
]

def read_if_exists(filename):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    if os.path.isfile(filepath):
        return open(filepath).read()
    else:
        ""

print('version: ', read_if_exists('version.txt') )

setup(
  name = 'DeepCL',
  version = read_if_exists('version.txt').strip(),
#  version = "3.4.0rc1",  # synchronize to deepcl main version
  author = "Hugh Perkins",
  author_email = "hughperkins@gmail.com",
  description = 'python wrapper for DeepCL deep convolutional neural network library for OpenCL',
  license = 'MPL',
  url = 'https://github.com/hughperkins/DeepCL',
  long_description = read('README.rst'),
  classifiers = [
    'Development Status :: 4 - Beta',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
  ],
  install_requires = [],
  # install_requires = [],
  tests_require = ['nose>=1.3.4','Cython>=0.22','cogapp>=2.4','future>=0.14.3'],
  scripts = ['test_deepcl.py','test_lowlevel.py'],
 # modules = libraries,
#  lib raries = libraries,
  ext_modules = my_cythonize( ext_modules),
)

