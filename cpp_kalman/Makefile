# https://stackoverflow.com/questions/2481269/how-to-make-a-simple-c-makefile
CXX := g++
# -g is for debug with gdb
# CPPFLAGS := -I c:\trash\eigen -std=c++17
CPPFLAGS := -I c:\work\eigen 

OBJS = main.o kalman.o dcm.o

kf: $(OBJS)
	$(CXX) -o kf $(OBJS)

debug: CPPFLAGS += -g
debug: kf

main.o: main.cpp

kalman.o: kalman.cpp kalman.h

dcm.o: dcm.h dcm.cpp


clean:
	del $(OBJS)
	del kf.exe
