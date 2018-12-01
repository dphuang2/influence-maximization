all : serial.o parallel.o

parallel.o : parallel.cu shared.cpp shared.h
	nvcc -std=c++11 -g -G -o parallel.o shared.cpp parallel.cu

serial.o : serial.cpp shared.cpp shared.h
	g++ -std=c++11 -g -o serial.o shared.cpp serial.cpp

clean:
	rm -rf *.o

