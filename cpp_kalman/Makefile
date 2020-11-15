# https://stackoverflow.com/questions/2481269/how-to-make-a-simple-c-makefile
CXX := g++
CC := g++
# -g is for debug with gdb
# CPPFLAGS := -I c:\trash\eigen -std=c++17
CPPFLAGS := -I/usr/include/eigen3

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


clean:
	del $(OBJS)
	del kf.exe
