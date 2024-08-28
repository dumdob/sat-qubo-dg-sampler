#pragma once
#include "config.hpp"

typedef vector<int> Vint;

template<size_t _N>
using bitvec = bitset<_N>;

template<size_t _N>
bool operator<(const bitvec<_N>& x, const bitvec<_N>& y)
{
    for (int i = 0; i < _N; i++) {
      if (x[i] ^ y[i]) return y[i];
    }
    return false;
}

typedef tuple<set<int>, bool, VectorXi> Cluster; //

template<size_t _N>
using VecInd = pair<int, bitvec<_N>>; //clID and the state

template<size_t _N>
struct VecIndCmp{
  bool operator()(const VecInd<_N> &v1, const VecInd<_N> &v2) const
  {
    for (int i = 0; i < _N; i++) {
        if (v1.second[i] ^ v2.second[i]) return v2.second[i];
    }
    return false;
  }
};

template<size_t _N>
struct VecCmp{
  bool operator()(const bitvec<_N> &v1, const bitvec<_N> &v2) const
  {
    for (int i = 0; i < _N; i++) {
        if (v1[i] ^ v2[i]) return v2[i];
    }
    return false;
  }
};


template<size_t _N>
bitvec<_N> tobitvec(const VectorXb &V)
{
  assert(V.size() == _N && "Dynamic vector to bitset is of wrong size!");

  bitvec<_N> v;
  for (int i = 0; i < _N; i++) {
    v[i] = V(i);
  }

  return v;
}

template<size_t _N>
VectorXb toVec(const bitvec<_N> &v)
{
  VectorXb V(v.size());

  for (int i = 0; i < _N; i++) {
    V(i) = v[i];
  }

  return V;
}

template<size_t _N>
using idset = set<VecInd<_N>, VecIndCmp<_N>>;

template<size_t _N>
using setmap = map<int, idset<_N>>;

template<size_t _N>
using E_set = pair<int, set<bitvec<_N>, VecCmp<_N>>>;

template<size_t _N>
using ordered_setvector = vector<E_set<_N>>;


inline int get_k_by_kID(const vector<pair<int, set<int>>>& Ek_vec, pair<int, int> EkID_key)
{
  for(int k = 0; k < Ek_vec.size(); k++){
    if(Ek_vec[k].first == EkID_key.first){
      auto search = Ek_vec[k].second.find(EkID_key.second);
      
      if(search != Ek_vec[k].second.end()){
        return k;
      }
    }
  }
  return -1;
}


template<size_t _N>
void setmap_to_ordered_setvector(setmap<_N> &sm,
                                const vector<pair<int, set<int>>>& Ek_vec,
                                ordered_setvector<_N> &sv_new)
{
  sv_new = ordered_setvector<_N>(Ek_vec.size());
  
  for(auto &s: sm)
  {
    for(auto iv = s.second.begin(); iv != s.second.end();)
    {
      int searchID = get_k_by_kID(Ek_vec, {s.first, iv->first});
      if(searchID != -1)
      {
        sv_new[searchID].first = s.first;
        sv_new[searchID].second.insert(iv->second);

        iv = s.second.erase(iv);
      }else{
        iv++;
      }
    }
  }
}

template <class I1, class I2>
bool have_common_element(I1 first1, I1 last1, I2 first2, I2 last2) {
    while (first1 != last1 && first2 != last2) {
        if (*first1 < *first2)
            ++first1;
        else if (*first2 < *first1)
            ++first2;
        else
            return true;
    }
    return false;
}

struct UboLandscape 
{
  MatrixXi M;
  int N0, N;

  // QUBO formulation
  Matrix<int, Dynamic, Dynamic, RowMajor> W;
  VectorXi B;
  int C;

  // PUBO formulation (3rd order)
  vector<SMatrixi> S0; 
  Matrix<int, Dynamic, Dynamic, RowMajor> W0;

  VectorXi B0;
  int C0;
  
  bool qubo_landscape = false;

  UboLandscape(const MatrixXi &M, int N0, 
    const MatrixXi &W, const VectorXi &B, int C,
    const vector<SMatrixi>& S0,
    const MatrixXi &W0, const VectorXi &B0, int C0);

  int Evalue(const VectorXb& Xb);
  int Evalue_native(const VectorXb& Xb);
  int dE_native(const VectorXb& Xb, int i);
  
  int dEqubo(const VectorXb &Xb, int xk, int Factor, int return_type);
  
  vector<E_set<_N0_>> random_descent(
    const VectorXb &X0, int *df_out, int rndseed = 0);

  void run_bfs(const set<bitvec<_N0_>, VecCmp<_N0_>>& states, int clID, idset<_N0_>& allclusters);

  void bfs_candidates(const bitvec<_N0_>& v, 
                      const set<bitvec<_N0_>, VecCmp<_N0_>>& bas,
                      const set<bitvec<_N0_>, VecCmp<_N0_>>& bfs_current,
                      set<bitvec<_N0_>, VecCmp<_N0_>>& bfs_new);
                      
  void bfs_candidates(const set<bitvec<_N0_>, VecCmp<_N0_>>& bfs_current, 
                      set<bitvec<_N0_>, VecCmp<_N0_>>& bfs_new, 
                      int clID,
                      idset<_N0_>& bas,
                      int& size_count);

  pair<ordered_setvector<_N0_>, MatrixXi> run_GWLBS(
    double beta,
    const VectorXd &E_partition, int K,
    int max_attempts,
    int rndseed,
    int BFS_E,
    bool run_final_bfs,
    const filesystem::path &out_pth, 
    ostream* logout);

};

void remove_top_minima(MatrixXi& barriers, int top);

template <typename Scalar>
void insert_cluster(int k, MatrixXi& barriers,
                    list<list<pair<VectorXt<Scalar>, Vint>>> &cross_states);

template <typename Scalar>
void unite_clusters(const vector<int> &ucv,
                    MatrixXi &barriers,
                    list<list<pair<VectorXt<Scalar>, Vint>>> &cross_states);
