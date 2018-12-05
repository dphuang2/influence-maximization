all : serial.o parallel.o

parallel.o : parallel.cu shared.cpp shared.h
	nvcc -std=c++11 -o parallel.o shared.cpp parallel.cu -O3 -Xptxas -O3

serial.o : serial.cpp shared.cpp shared.h
	g++ -std=c++11 -o serial.o shared.cpp serial.cpp -O3

clean:
	rm -rf *.o

