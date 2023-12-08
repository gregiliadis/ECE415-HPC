CC=nvcc
CFLAGS=-O4

all: optimized tile stream manipulator

%.o: %.cu
	$(CC) -o $@ $< $(CFLAGS)

%.o: %.c
	gcc -o $@ $< -lm

optimized: optimized.o
	$(CC) $(CFLAGS) optimized.cu -o optimized

tile: tile.o
	$(CC) $($CFLAGS) tile.cu -o tile

stream: stream.o
	$(CC) $(CFLAGS) stream.cu -o stream

manipulator: manipulator.o
	gcc -Wall -g  manipulator.c -o manipulator -lm
clean:
	rm *.o optimized tile stream manipulator
