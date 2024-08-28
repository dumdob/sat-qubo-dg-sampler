#include "mcmc.hpp"

#include <vector>
#include <set>
#include <numeric>
#include <algorithm>
#include <random>
#include <iterator>
#include <iostream>

using std::vector;
using std::set;
using std::pair;
using std::cout;
using std::endl;

#define MH_SAMPLING

#include <Eigen/Dense>

using Eigen::VectorXd;
using Eigen::ArrayXd;
using Eigen::VectorXi;
using Eigen::seq;

//row major for correct alignment with numpy
using MatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixXi = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

struct Factors{
  vector<vector<int>> indices;
  VectorXi values;
};

int run_energy_sat(
        int* x_start, int x_start_size,
        int* J_indices, int J_indices_size,
        int* J_values, int J_values_size,
        int* h, int h_size)
{
  int N = x_start_size;
  VectorXi _x  = Eigen::Map<VectorXi>(x_start, N);
  VectorXi _h  = Eigen::Map<VectorXi>(h, N);

  int e = -(_x.transpose()*_h)[0];

  VectorXi _J_values  = Eigen::Map<VectorXi>(J_values, J_values_size);
  VectorXi read_indices = Eigen::Map<VectorXi>(J_indices, J_indices_size);

  int a = 0;
  int m = _J_values(0);
  for(auto it: read_indices){  
    if (it == -1){
      e -= m;
      a++;
      m = _J_values(a);
    }else
    {
      m *= _x(it);
    }
  }

  return e;
}

inline int local_field(const VectorXi& x,
                       const Factors& J, const VectorXi& h,
                       const vector<set<int>>& i_to_factors, int k)
{
  int d = h(k);
  for(auto a: i_to_factors[k])
  {
    int m = J.values[a];
    for(auto it: J.indices[a])
    {
      if(it != k)
        m *= x(it);
      
    }
    d += m;
  }
  return d;
}

vector<double> MCMC(VectorXi& final_x, 
                    VectorXi& min_x,
                    const VectorXi& x_start,
                    const std::string mode,
                    const Factors& J, const VectorXi& h,
                    const vector<set<int>>& i_to_factors,
                    int seed, 
                    int total_sweeps,
                    double initial_T,
                    double final_T,
                    const std::string T_schedule,
                    int aux_size, int track_stats_freq)
{
  int N = x_start.size();
  int native_size = N - aux_size;
  
  int Nn = N;

  VectorXi x = x_start;
  min_x = x;
  
  //compute starting energy
  int e = -(x.transpose()*h)[0];

  for (size_t a = 0; a < J.values.size(); a++)
  {
    int m = J.values[a];
    for (auto i: J.indices[a]){
      m *= x(i);
    }
    e -= m;
  }
  int min_e = e;

  VectorXd temp_v(total_sweeps);

  if(initial_T == 0)
    initial_T = 1e-5;

  if(final_T == 0)
    final_T = 1e-5;

  if (T_schedule == "linear")
  {
    temp_v = VectorXd::LinSpaced(total_sweeps, initial_T, final_T);

  }else if (T_schedule == "exponential")
  {
    if(initial_T == final_T)
    {
      temp_v = VectorXd::Constant(total_sweeps, initial_T);

    }else{
      double tau = (total_sweeps-1)/std::log(initial_T/final_T);

      for (int i = 0; i < total_sweeps; i++)
      {
        temp_v(i) = initial_T*exp(-1.0*i/tau);
      }
    }

  }else{
    throw std::runtime_error("Wrong temperature schedule!");
  }

  vector<int> perm(Nn);
  std::iota(perm.begin(), perm.end(), 0);
  
  std::mt19937 g;
  std::mt19937 g2;
  
  g.seed(seed);
  g2.seed(seed);

  std::uniform_real_distribution<> dis(0, 1);

  vector<double> ar_trace;
  double AR_sum = 0; //acceptance rate

  for (size_t j = 0; j < total_sweeps; j++)
  {
    double temp = temp_v(j);
    std::shuffle(perm.begin(), perm.end(), g2);

    for (size_t k = 0; k < Nn; k++)
    {
      int kk = perm[k];
      int dx = (2*x(kk)-1);
      
      int d = local_field(x, J, h, i_to_factors, kk);
      
      auto g_save = dis(g);
      #ifdef MH_SAMPLING
        int xtmp = x(kk);
        if (g_save < std::exp(-d*dx/temp))
          xtmp = 1 - x(kk);
        
      #else
        int signbit = std::signbit(std::tanh(0.5*d/temp) - 2 * g_save + 1);
        int xtmp = 1 - signbit;

      #endif
      
      if(x(kk) != xtmp)
      {
        AR_sum++;

        e += d*dx;
        x(kk) = xtmp;

        if(e < min_e){
          min_e = e;
          min_x = x;
        }
      }
    }

    if((j+1) % track_stats_freq == 0)
    {
      ar_trace.push_back(AR_sum/Nn/track_stats_freq);
      AR_sum = 0;
    }
  }
  
  final_x = x;

  return ar_trace;
}


void run_mcmc_sat(
        int final_x_size, int* final_x,
        int min_x_size, int* min_x,

        int ar_trace_size, double* ar_trace,

        int* x_start, int x_start_size,

        char * mode,
        
        int* J_indices, int J_indices_size,
        int* J_values, int J_values_size,

        int* h, int h_size,
        
        int seed, int total_sweeps,
        double initial_T, 
        double final_T,

        char * T_schedule,

        int aux_size, int track_stats_freq)
{
  int N = x_start_size;

  int native_size = N - aux_size;

  VectorXi _x_start = Eigen::Map<VectorXi>(x_start, N);

  VectorXi _final_x(N);
  VectorXi _min_x(N);

  VectorXi _h = Eigen::Map<VectorXi>(h, N);
  Factors _J;
  _J.values  = Eigen::Map<VectorXi>(J_values, J_values_size);

  VectorXi read_indices = Eigen::Map<VectorXi>(J_indices, J_indices_size);

  vector<set<int>> i_to_factors(N);
  vector<int> idx;
  for(auto it: read_indices){  
    if (it != -1){
      idx.push_back(it);
      i_to_factors[it].insert(_J.indices.size());

    }else{
      _J.indices.push_back(idx);
      idx.clear();
    }
  }
  
  vector<double> mcmc_ar_trace = MCMC(_final_x, _min_x, 
                                      _x_start, 

                                      mode,

                                      _J, _h, i_to_factors, 

                                      seed, 
                                      total_sweeps,
                                      initial_T, final_T, T_schedule,

                                      aux_size, track_stats_freq);

  for (size_t i = 0; i < ar_trace_size; i++){
    ar_trace[i] = mcmc_ar_trace[i];
  }

  for (size_t i = 0; i < N; i++)
  {
    final_x[i] = _final_x(i);
    min_x[i] = _min_x(i);
  }
}
