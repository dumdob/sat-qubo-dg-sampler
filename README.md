# Disconnectivity graph sampler for PUBO/QUBO mappings of 3-SAT problems

C++ code used to generate results of the paper 
[Disconnectivity graphs for visualizing combinatorial optimization problems: challenges of embedding to Ising machines (arXiv:2403.01320)](https://arxiv.org/abs/2403.01320).

## Build
### Compiler and Libraries
A C++17 and OpenMP (optional) compatible compiler. 

Libraries required:
- [Eigen C++ template library 3.4.*](https://eigen.tuxfamily.org/dox/)
- [JSON for Modern C++](https://github.com/nlohmann/json)
- [GSL - GNU Scientific Library](https://www.gnu.org/software/gsl/)

### Makefile

Modify makefile to support the aforementioned compilers and libraries on your system and run `make`.

To change the internal configuration of the sampler, modify the file `config.hpp` accordingly (see also comments for more details).

## Config
`config.json` file contains the parameters that can be changed without recompiling. They include: number of parallel threads (if OpenMP is supported), output directory name, random seed, problem class, maximum GWL number of steps, saving all local minima states (true/false).

## Run
To run the sampler, the following command-line parameters need to be specified: problem instance number, mapping ID (0 for PUBO, 1 for Rozenberg, 2 for KZFD), run ID (arbitrary).

```
./dg_sampler.out 1 0 testrun
```

Several files with the given run ID name will be generated in the output directory: log file with all details of the run for each parallel thread, energy barriers matrix, local minimum/saddle cluster degeneracy, and the GWL historgram for each of the samplers in i-th parallel thread (see also comments in `config.hpp` for more details).

## Disconnectivity graphs construction
To construct disconnectivity graphs from the sampled data one could use open-source library [pele: Python Energy Landscape Explorer](http://pele-python.github.io/pele/)