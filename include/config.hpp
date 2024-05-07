#pragma once

#include <iostream>
#include <fstream>
#include <list>
#include <map>
#include <set>
#include <bitset>
using namespace std;

#ifdef _OPENMP
#include "omp.h"
#endif

#include <filesystem>
using pth = filesystem::path;

#include <Eigen/Dense>
#include <Eigen/Sparse>
using idx = Eigen::Index;

using namespace Eigen;

typedef SparseMatrix<int> SMatrixi;
typedef Triplet<int> Ti;

typedef Matrix<bool, Dynamic, 1> VectorXb;

template <typename Scalar>
using ReadMatrix = Matrix<Scalar, Dynamic, Dynamic, RowMajor>;

template <typename Scalar>
using VectorXt = Matrix<Scalar, Dynamic, 1>;

#include <iterator>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

enum class MapIDs{
  PUBO = 0,
  Rosenberg_advanced,
  KZFD_advanced
};

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#define _N0_ 50               //Problem size
#define _QUBO_BAR_FACTOR_ 2   //QUBO barrier factor
// #define _WRITE_COUT_       //Write log to cout (not file)

// #define _BALLISTIC_SEARCH_ 
#define _RS_LIMIT_ 20         //random search in the plateu region limit

#define _GWL_BFS_LIMIT_ 500   //BFS limit per cluster during sampling
#define _CUMULATIVE_BFS_LIMIT_ 10000000 //BFS limit per energy during postprocessing

#define _RESTART_LIMIT_ 10000 //maximum steps allowed in a single basin before random restart
#define _HIGH_E_LIMIT_  10000 //maximum steps allowed in high E region before annealing

// Output of the program:
// - energy barrier matrix; symmetric, local minimum/saddle energy on the diagonal, -1 stands for the undiscovered barrier between corresponding local minima/saddles
// - GWL histogram; saddles have 0 visits, last column stands for visits to the untracked basins, last row denotes local minimum/saddle type of a degenerate cluster (1/0)
// - cluster degeneracies in the corresponding order of the barrier matrix
// - explicit local minima states in the corresponding order of the barrier matrix