
CPP = clang++-3.5
#CPP = g++

CPP_FLAGS = -O3 -std=c++11 -I./include

main: build/Node.o build/DecisionTree.o build/common.o build/RandomForest.o
	$(CPP) $(CPP_FLAGS)    -o main main.cpp build/*.o

build/Node.o:
	$(CPP) $(CPP_FLAGS) -c -o build/Node.o src/Node.cpp

build/DecisionTree.o:
	$(CPP) $(CPP_FLAGS) -c -o build/DecisionTree.o src/DecisionTree.cpp

build/common.o:
	$(CPP) $(CPP_FLAGS) -c -o build/common.o src/common.cpp

build/RandomForest.o:
	$(CPP) $(CPP_FLAGS) -c -o build/RandomForest.o src/RandomForest.cpp
