EXE=videofilter
CVINCFLAGS=`pkg-config --libs --cflags opencv`
CVLIBFLAGS=`pkg-config --libs opencv`
OCLLIBSDIR=/opt/ComputeLibrary/build/
OCLINCSDIR=/opt/ComputeLibrary/include/
LDFLAGS=-L${OCLLIBSDIR} -larm_compute -larm_compute_core -lOpenCL
DBGFLAGS= -pg -g
FLAGS= -std=gnu++11
GCC=g++

SRCS=./videofilter_gpu.cpp
#SRCS=./videofilter_cpu.cpp

all:${EXE}
${EXE}: ${SRCS}
	${GCC} ${DBGFLAGS}  ${CVINCFLAGS} ${SRCS} ${CVLIBFLAGS} -o ${EXE} ${LDFLAGS}


profile:
	./${EXE}
	gprof ./${EXE}  gmon.out > prof.txt
clean:
	rm -rf ${EXE} *.o
