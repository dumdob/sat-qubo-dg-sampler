#pragma once


//====== Configure for your problem class ======
#define _N0_ 50               //Native problem size
#define _QUBO_BAR_FACTOR_ 1   //QUBO barrier factor (for QUBO only)
#define _WRITE_COUT_       //Write log to cout (not file)

#define _BETA_SCALE_ 1.0      //beta = 1/T = _BETA_SCALE_/<dE> hyperparameter

#define _RS_LIMIT_ 20         //random search in the plateu region limit

#define _GWL_BFS_LIMIT_ 10             //BFS limit per cluster during sampling
#define _CUMULATIVE_BFS_LIMIT_ 10000000 //BFS limit per energy level during postprocessing

#define __RESTARTS_PERIOD__ 20000 //step period of restarts if stuck in high E region or a specific local minimum
#define __MAX_RESTARTS__    1000  //maximum number of restarts if stuck in high E region or a specific local minimum



//====== Output of the program ======
// - energy barrier matrix; symmetric, local minimum/saddle energy on the diagonal, -1 stands for an undiscovered barrier between corresponding local minima/saddles
// - GWL histogram; saddles have 0 visits, last column stands for visits to the untracked basins, last row denotes local minimum/saddle type of a degenerate cluster (0/1)
// - cluster degeneracies (number of stable connected states) in the corresponding order of the barrier matrix
// - explicit local minimum states in the corresponding order of the barrier matrix
// - logs



#include <iostream>
#include <fstream>
#include <list>
#include <map>
#include <set>
#include <bitset>
#include <chrono>
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
