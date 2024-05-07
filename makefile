INC_DIR = include

SRC_DIR = src
BUILD_DIR = build

SRC_FILES  = $(wildcard $(SRC_DIR)/*.cpp)

OBJ_FILES  = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRC_FILES))

INC = $(patsubst %, -I%, $(INC_DIR))

DBG_FLAGS = -DEIGEN_DONT_PARALLELIZE -O3 -w -g -DNDEBUG -DEIGEN_NO_DEBUG

CXX = clang++
OMP_FLAGS = -fopenmp
CXXFLAGS = -std=c++17 $(DBG_FLAGS) $(OMP_FLAGS)

LPATHS = 
LFLAGS = gsl gslcblas omp
LIB = $(patsubst %, -L%, $(LPATHS)) $(patsubst %, -l%, $(LFLAGS))

all: dg_sampler.out

dg_sampler.out: $(OBJ_FILES)
	$(CXX) $(LIB) $(OBJ_FILES) -g -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(@D)
	$(CXX) -c $(CXXFLAGS) $< -o $@ $(INC)
	@$(CXX) -MM $< -MP -MT $@ -MF $(@:.o=.d) $(INC)

.PHONY: all cleanup

cleanup:
	rm -fr $(BUILD_DIR)
	rm -f  dg_sampler.out

-include $(OBJ_FILES:.o=.d)
