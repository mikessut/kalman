# https://stackoverflow.com/questions/2481269/how-to-make-a-simple-c-makefile
CXX := g++
# -g is for debug with gdb
CPPFLAGS := -I c:\trash\eigen -g -std=c++17

OBJS = main.o kalman.o dcm.o

kf: $(OBJS)
	$(CXX) -o kf $(OBJS)

main.o: main.cpp

kalman.o: kalman.cpp kalman.h

dcm.o: dcm.h dcm.cpp


clean:
	del $(OBJS)
