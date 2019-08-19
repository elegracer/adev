all: ./allan_deviation.h ./allan_deviation.cpp ./gnuplot.h ./random.h ./ransac.h
	g++ allan_deviation.cpp -O3 -std=c++17 -pthread -o adev

rm:
	rm *.o adev
