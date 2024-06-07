exec: main.o 
	g++ $(CFLAGS) -o exec main.o 
	
main.o: main.cpp main.h
	g++ -std=gnu++0x -g -c main.cpp


rem:
	rm *.o;	rm exec; rm -r Debug

