CC=nvcc
CFLAGS=-O4

all: conv convG convGdoubles convP manipulator

%.o: %.cu
	$(CC) -o $@ $< $(CFLAGS)

conv: Convolution2D.o
	$(CC) $(CFLAGS) Convolution2D.cu -o conv

convG: Convolution2DGrid.o
	$(CC) $(CFLAGS) Convolution2DGrid.cu -o convG

convGdoubles: Convolution2DGrid_doubles.o
	$(CC) $($CFLAGS) Convolution2DGrid_doubles.cu -o convGdoubles

convP: Convolution2DPadd.o
	$(CC) $(CFLAGS) Convolution2DPadd.cu -o convP

manipulator: manipulator.o
	$(CC)  manipulator.c -o manipulator
clean:
	rm *.o conv convG convGdoubles convP manipulator
