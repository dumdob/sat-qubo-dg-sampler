INC_DIR = include $(CONDA_PREFIX)/include/eigen3 $(CONDA_PREFIX)/include

SRC_DIR = src
BUILD_DIR = build

SRC_FILES  = $(wildcard $(SRC_DIR)/*.cpp)

OBJ_FILES  = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRC_FILES))

INC = $(patsubst %, -I%, $(INC_DIR))

DBG_FLAGS = -DEIGEN_DONT_PARALLELIZE -O3 -w -g -DNDEBUG -DEIGEN_NO_DEBUG

CXX = g++
OMP_FLAGS = -fopenmp
CXXFLAGS = -std=c++17 $(DBG_FLAGS) $(OMP_FLAGS)

LPATHS = $(CONDA_PREFIX)/lib
LFLAGS = gsl gslcblas m gomp
LIB = $(patsubst %, -L%, $(LPATHS)) $(patsubst %, -l%, $(LFLAGS))

all: dg_sampler.out

dg_sampler.out: $(OBJ_FILES)
	$(CXX) -static -o $@ $(OBJ_FILES) $(LIB) 

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(@D)
	$(CXX) -c $(CXXFLAGS) $< -o $@ $(INC)
	@$(CXX) -MM $< -MP -MT $@ -MF $(@:.o=.d) $(INC)

.PHONY: all cleanup

cleanup:
	rm -fr $(BUILD_DIR)
	rm -f  dg_sampler.out

-include $(OBJ_FILES:.o=.d)
