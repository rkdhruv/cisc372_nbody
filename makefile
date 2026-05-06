NVCC = nvcc
CC = gcc
FLAGS = -DDEBUG
NVCCFLAGS = $(FLAGS)
LIBS = -lm
ALWAYS_REBUILD = makefile

nbody: nbody.o compute.o
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LIBS)
nbody.o: nbody.c planets.h config.h vector.h compute.h $(ALWAYS_REBUILD)
	$(CC) $(FLAGS) -c $<
compute.o: compute.cu config.h vector.h compute.h $(ALWAYS_REBUILD)
	$(NVCC) $(NVCCFLAGS) -c $<
clean:
	rm -f *.o nbody
