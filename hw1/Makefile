CXX = g++
CXXFLAGS=-std=c++11 -O3 -march=native -Wall -I$(SRC_DIR) $(shell python3 -m pybind11 --includes)
PY_LDFLAGS=$(shell python3-config --ldflags) -lopenblas -shared -fPIC
LDFLAGS=-lopenblas
SRC_DIR=pupok/src
PYTHON_DIR=pupok/python

pupok: $(PYTHON_DIR)/bindings.o $(SRC_DIR)/cosine_similarity.o
	$(CXX) $^ -o $(PYTHON_DIR)/pupok_core`python3-config --extension-suffix` $(PY_LDFLAGS) $(CXXFLAGS)

$(PYTHON_DIR)/bindings.o: $(PYTHON_DIR)/bindings.cpp $(SRC_DIR)/cosine_similarity.h
	$(CXX) $(CXXFLAGS) -fPIC -c $< -o $@

$(SRC_DIR)/cosine_similarity.o: $(SRC_DIR)/cosine_similarity.cpp $(SRC_DIR)/cosine_similarity.h
	$(CXX) $(CXXFLAGS) -fPIC -c $< -o $@


clean:
	rm -f $(PYTHON_DIR)/*.o $(SRC_DIR)/*.o $(TESTS_DIR)/*.o $(PYTHON_DIR)/pupok_core`python3-config --extension-suffix`