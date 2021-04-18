# https://stackoverflow.com/questions/2481269/how-to-make-a-simple-c-makefile
CXX := g++
CC := g++
# -g is for debug with gdb
# CPPFLAGS := -I c:/work/eigen -I. -std=c++17
CPPFLAGS := -I/usr/include/eigen3 -I.   # -g

BUILD_DIR = build
TEST_DIR = tests

#OBJS = main.o kalman.o dcm.o pressure.o kalman_euler.o
#OBJS = kalman_euler.o kalman.o
OBJS = kalman.o
TESTS = test_heading.o

#kf: $(OBJS)
#	$(CXX) -o kf $(OBJS)

# debug: CPPFLAGS += -g
# debug: kf

$(BUILD_DIR)/%.o: %.cpp
	$(CXX) -c $(CPPFLAGS) $< -o $@

$(BUILD_DIR)/%.o: $(TEST_DIR)/%.cpp
	$(CXX) -c $(CPPFLAGS) $< -o $@

kf: $(OBJS)
	$(CXX) -o kf $(OBJS)

tests: $(BUILD_DIR)/$(OBJS) $(BUILD_DIR)/$(TESTS)
	$(CXX) -o $(BUILD_DIR)/tests $(BUILD_DIR)/$(OBJS) $(BUILD_DIR)/$(TESTS)

$(BUILD_DIR)/$(TESTS)/%.o:
	echo $@
	echo $%
	echo $<
	echo $?
	echo $^
	echo $+


#kalman_pic.o: kalman.cpp
#	$(CXX) $(CPPFLAGS) -c -fpic kalman.cpp -o kalman_pic.o

#python: kalman_pic.o
#	$(CXX) $(CPPFLAGS) -c -fpic kalman_python_wrapper.cpp
#	$(CXX) -shared -o libkalman.so kalman_python_wrapper.o kalman_pic.o

# clean:
# 	rm -f $(OBJS)
# 	rm -f kalman_pic.o
# 	rm -f libkalman.so
# 	rm -f kf

