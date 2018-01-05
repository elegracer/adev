all: ./allan_deviation.h ./allan_deviation.cpp ./gnuplot.h ./Random.h ./RANSAC.h
	g++ allan_deviation.cpp -Ofast -std=c++17 -lstdc++fs -pthread -o adev

rm:
	rm *.o adev

