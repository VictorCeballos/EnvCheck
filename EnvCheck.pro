################################################################################
# Victor Ceballos Inza
################################################################################

TARGET = EnvCheck
TEMPLATE = app

# Project build directories
DESTDIR = $$OUT_PWD #$$system(pwd)
OBJECTS_DIR = $$OUT_PWD
message(build dir = $$DESTDIR)

# Config
CONFIG += console c++11
CONFIG -= app_bundle

# Qt
QT += core gui opengl
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets
QMAKE_CXXFLAGS_RELEASE = -O3

################################################################################
# LIBS
################################################################################

#
LIBS += -fopenmp

# OpenGL
LIBS += -lGL -lGLU -lGLEW -lglut

# OpenCL
LIBS += -lOpenCL

# OpenSceneGraph
LIBS += -losg -losgGA -losgDB -losgViewer -losgQt -losgUtil -losgManipulator
LIBS += -lOpenThreads

# CUDA
LIBS += -L$$CUDA_DIR/lib64 -lcudart -lcuda

################################################################################
# CUDA
################################################################################

QMAKE_EXTRA_COMPILERS += cuda

# CUDA variables
CUDA_DIR        = /usr/local/cuda
INCLUDEPATH    += $$CUDA_DIR/include
INCLUDEPATH    += $$CUDA_DIR/samples/common/inc
QMAKE_LIBDIR   += $$CUDA_DIR/lib64
QMAKE_RPATHDIR += $$CUDA_DIR/lib64
CUDA_LIBS      += -lcudart -lcuda
CUDA_INC        = $$join(INCLUDEPATH, ' -I', '-I', ' ')
CUDA_ARCH       = sm_20

# NVCC flags
NVCCFLAGS = --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v

# Prepare the extra compiler configuration
# nvcc error printout format ever so slightly different from gcc
# http://forums.nvidia.com/index.php?showtopic=171651
cuda.commands = $$CUDA_DIR/bin/nvcc -m64 -O3 -arch=$$CUDA_ARCH -c $$NVCCFLAGS \
                $$CUDA_INC $$CUDA_LIBS ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT} \
                2>&1 | sed -r \"s/\\(([0-9]+)\\)/:\\1/g\" 1>&2
cuda.dependency_type = TYPE_C
cuda.depend_command = $$CUDA_DIR/bin/nvcc -O3 -M $$CUDA_INC $$NVCCFLAGS ${QMAKE_FILE_NAME}
cuda.input = CUDA_SOURCES
cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.o

################################################################################
# Soruces
################################################################################

HEADERS += \
    inc/TestCUDA.h \
    inc/TestOpenCL.h \
    inc/TestOpenGL.h

SOURCES += \
    main.cpp \
    src/TestCUDA.cpp \
    src/TestOpenCL.cpp \
    src/TestOpenGL.cpp

################################################################################
# CUDA Sources
################################################################################

CUDA_SOURCES += \
    src/cuda_code.cu

