## Build
Libraries required:
- Python with Numpy, JSON libraries
- SWIG interface generator

To setup the simulated annealing C++ extension interface, run swig command in the c_utils folder:

```
swig -c++ -python c_interface.i
```

To compile the C++ extension, modify setup.py for your system and run:

```
python setup.py build_ext --inplace
```

## Config
General configuration of the solver is controlled by `configs/default_config.json`; 
specific configurations are provided by each problem class in the corresponding folders, 
i.e. `configs/[problem_class]/config.json`.

## Run
Run `main.py`. 
Upon completion, program outputs the number of successful runs for each 
problem instance, as well as the time-to-solution statistics.