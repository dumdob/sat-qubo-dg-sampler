# Disconnectivity graphs sampler + simulated annealing for PUBO/QUBO mappings of 3-SAT problems

C++ code used to generate results of the paper: 
[Energy landscapes of combinatorial optimization in Ising machines (arXiv:2403.01320)](https://arxiv.org/abs/2403.01320).

## Build
### Compiler and Libraries
A C++17 and OpenMP (optional) compatible compiler. 

Libraries required:
- [Eigen C++ template library 3.4.*](https://eigen.tuxfamily.org/dox/)
- [JSON for Modern C++](https://github.com/nlohmann/json)
- [GSL - GNU Scientific Library](https://www.gnu.org/software/gsl/)

### Makefile

Modify makefile to support the aforementioned compilers and libraries on your system and run `make`.
To change the internal configuration of the sampler for your problem class, modify the file `include/config.hpp` accordingly (see also comments for more details).

## Config
`config.json` file contains the parameters that can be changed without recompiling. They include: number of parallel threads (if OpenMP is supported), output directory name, run name, random seed, problem class, maximum GWL number of steps, mapping ID (0 for PUBO, 1 for Rozenberg, 2 for KZFD), number of tracked clusters (K), maximum energy of GWL histogram, maximum energy of breadth-first search, saving all local minima states (true/false).

## Problem instances
Problem instances need to be stored in the "problems/[problem_class]" directory; 
3-SAT instances should be named as [problem_class][problem_size]-0[instance_number].cnf

## Run
Run the program specifying the [instance_number] number
```
./dg_sampler.out [instance_number]
```

## Output
Several files with the given run name will be generated in the output directory: log file with all details of the run for each parallel thread, energy barriers matrix, local minimum/saddle cluster degeneracy, and the GWL historgram for each of the samplers in parallel threads.
