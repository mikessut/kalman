# https://stackoverflow.com/questions/2481269/how-to-make-a-simple-c-makefile
CXX := g++
CC := g++
# -g is for debug with gdb
# CPPFLAGS := -I c:\trash\eigen -std=c++17
CPPFLAGS := -I/usr/include/eigen3 -g

OBJS = main.o kalman.o dcm.o pressure.o

kf: $(OBJS)
	$(CXX) -o kf $(OBJS)

debug: CPPFLAGS += -g
debug: kf

main.o: main.cpp

kalman.o: kalman.cpp kalman.h

dcm.o: dcm.h dcm.cpp

pressure.o: pressure.h pressure.c

scratch.o: scratch.cpp

scratch: scratch.o

kalman_pic.o: kalman.cpp
	$(CXX) $(CPPFLAGS) -c -fpic kalman.cpp -o kalman_pic.o

python: kalman_pic.o
	$(CXX) $(CPPFLAGS) -c -fpic kalman_python_wrapper.cpp
	$(CXX) -shared -o libkalman.so kalman_python_wrapper.o kalman_pic.o

clean:
	rm -f $(OBJS)
	rm -f kalman_pic.o
	rm -f libkalman.so
	rm -f kf
