/*-----------------------------------------------------------------------------

Simulated annealer swig interface

-----------------------------------------------------------------------------*/
%module c_interface
%{
/* Includes the header in the wrapper code */
#define SWIG_FILE_WITH_INIT
#include "mcmc.hpp"
%}

%include "numpy.i"

%init %{
import_array();
%}


%apply (int DIM1, int* ARGOUT_ARRAY1) { (int final_x_size, int* final_x)};
%apply (int DIM1, int* ARGOUT_ARRAY1) { (int min_x_size, int* min_x)};

%apply (int DIM1, double* ARGOUT_ARRAY1) { (int ar_trace_size, double* ar_trace)};

%apply (int* IN_ARRAY1, int DIM1) { (int* x_start, int x_start_size)};

%apply (int* IN_ARRAY1, int DIM1) { (int* J_indices, int J_indices_size)};
%apply (int* IN_ARRAY1, int DIM1) { (int* J_values, int J_values_size)};
%apply (int* IN_ARRAY1, int DIM1) { (int* h, int h_size)};

/* Parse the header file to generate wrappers */
%include "mcmc.hpp"
