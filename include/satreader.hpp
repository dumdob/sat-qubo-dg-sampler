#pragma once

#include "config.hpp"

inline idx read_cnf(MatrixXi &M, string path)
{
  // The cnf file should begin with "p cnf"
  std::ifstream fin(path);
  std::string skip;
  getline(fin, skip, 'f');

  std::istream_iterator<int> start_f(fin), end_f;
  std::vector<int> cnf_list(start_f, end_f);

  idx Nvars = cnf_list[0];
  cnf_list.erase(cnf_list.begin());

  idx Nclauses = cnf_list[0];
  cnf_list.erase(cnf_list.begin());

  M = Map<ReadMatrix<int>>(cnf_list.data(), Nclauses , 4);
  M = M(all, seq(0, 2)).eval(); // cnf is read to a N_clauses x 3 matrix

  return Nvars;
}

inline MatrixXi QUBO_advanced(MatrixXi &W, 
                                   VectorXi &B, 
                                   int &C, 
                                   string path,                                
                                   bool KZFD)
{
  MatrixXi M;
  idx Nvars = read_cnf(M, path);
  idx Nclauses = M.rows();

  M *= -1;

  MatrixXi W1 = MatrixXi::Zero(Nvars, Nvars);
  MatrixXi B1 = VectorXi::Zero(Nvars);

  C = 0;

  MatrixXi Midx = M.cwiseAbs() - MatrixXi::Constant(M.rows(), M.cols(), 1);

  for (size_t i = 0; i < Nclauses; i++)
  {
    if(M(i, 2) == 0)
    {      
      throw runtime_error("2SAT support!");
    }

    if (M(i,0) > 0) {
      if (M(i,1) > 0) {
        if (M(i,2) > 0) {
          ;
        }else{
          W1(Midx(i,0), Midx(i,1)) += 1;
        }
      }else{
        if (M(i,2) > 0) {
          W1(Midx(i,0), Midx(i,2)) += 1;
        }else{
          W1(Midx(i,0), Midx(i,1)) -= 1;
          W1(Midx(i,0), Midx(i,2)) -= 1;

          B1(Midx(i,0)) += 1;
        }
      }
    }else{
      if (M(i, 1) > 0) {
        if (M(i,2) > 0) {
          W1(Midx(i,1), Midx(i,2)) += 1;
        }else{
          W1(Midx(i,0), Midx(i,1)) -= 1;
          W1(Midx(i,1), Midx(i,2)) -= 1;

          B1(Midx(i,1)) += 1;
        }
      }else{
        if (M(i,2) > 0) {
          W1(Midx(i,0), Midx(i,2)) -= 1;
          W1(Midx(i,1), Midx(i,2)) -= 1;

          B1(Midx(i,2)) += 1;
        }else{
          W1(Midx(i,0), Midx(i,1)) += 1;
          W1(Midx(i,0), Midx(i,2)) += 1;
          W1(Midx(i,1), Midx(i,2)) += 1;

          B1(Midx(i,0)) -= 1;
          B1(Midx(i,1)) -= 1;
          B1(Midx(i,2)) -= 1;

          C += 1; //add the constant
        }
      }
    }
  }
  

  VectorXi count3 = VectorXi::Constant(Nclauses, 1);
  
//  Getting rid of extra 3rd order terms
  for (size_t i = 0; i < Nclauses; i++)
  {
    if (count3(i) == 0){
      //skip already matched clauses
      continue;
    }
    //sort the variables in a clause
    RowVector3i v1 {Midx(i,0), Midx(i,1), Midx(i,2)}; sort(v1.begin(), v1.end());
    
    bool first_match = true;
    for (size_t j = i+1; j < Nclauses; j++)
    {
      if (count3(j) == 0){
        //skip already matched clauses
        continue;
      }
      //sort the variables in a clause
      RowVector3i v2 {Midx(j,0), Midx(j,1), Midx(j,2)}; sort(v2.begin(), v2.end());
      if(v1 == v2)
      {
        if (first_match){
          if (M(i,0)*M(i,1)*M(i,2) > 0){
            count3(i) = 1;
          }else{
            count3(i) = -1;
          }
          first_match = false;
        }
        
        if (M(j,0)*M(j,1)*M(j,2) > 0){
          count3(i) += 1;
        }else{
          count3(i) -= 1;
        }
        
        count3(j) = 0;
      }
    }
    
  }
  
  MatrixXi A = MatrixXi::Zero(Nvars, Nvars);
  for (size_t i = 0; i < Nclauses; i++)
  {
    //don't consider vars from extra repeating clauses
    if(count3(i) != 0)
    {
      A(Midx(i,0), Midx(i,1))++;
      A(Midx(i,0), Midx(i,2))++;
      A(Midx(i,1), Midx(i,2))++;
    }
  }
  A += A.transpose().eval();

  vector<pair<Vector2i, int>> vpairs;

  for (size_t i = 0; i < A.rows(); i++) {
    for (size_t j = i+1; j < A.cols(); j++)
    {
      if (A(i,j) > 0) {
        Vector2i v{i, j}; sort(v.begin(), v.end());
        vpairs.push_back(make_pair(v, A(i,j)));
      }
    }
  }

  stable_sort(vpairs.begin(), vpairs.end(),
    [](const pair<Vector2i, int> &a, const pair<Vector2i, int> &b){
      return a.second > b.second;
    }
  );

  vector<Vector2i> vpairsW;

  VectorXi track_clauses = VectorXi::Constant(Nclauses, 1);
  
  MatrixXi Midx2 = Midx;
  for (size_t i = 0; i < Nclauses; i++)
  {
    //skip already considered clauses
    if(track_clauses(i) == 0){
      continue;
    }
    
    if(count3(i) == 0){
      continue;
    }
    
    Vector2i v01 {Midx2(i,0), Midx2(i,1)}; sort(v01.begin(), v01.end());
    Vector2i v02 {Midx2(i,0), Midx2(i,2)}; sort(v02.begin(), v02.end());
    Vector2i v12 {Midx2(i,1), Midx2(i,2)}; sort(v12.begin(), v12.end());

    for (size_t k = 0; k < vpairs.size(); k++)
    {
      bool pushpair = false;

      if      (v01 == vpairs[k].first ) {
        pushpair = true;
        for (size_t l = 0; l < vpairs.size(); l++)
          if(v02 == vpairs[l].first || v12 == vpairs[l].first){
            vpairs[l].second--;
          }
      }
      else if (v02 == vpairs[k].first) {
        pushpair = true;
        for (size_t l = 0; l < vpairs.size(); l++)
          if(v01 == vpairs[l].first || v12 == vpairs[l].first){
            vpairs[l].second--;
          }
      }
      else if (v12 == vpairs[k].first) {
        pushpair = true;
        for (size_t l = 0; l < vpairs.size(); l++)
          if(v01 == vpairs[l].first || v02 == vpairs[l].first){
            vpairs[l].second--;
          }
      }

      if (pushpair) {
        vpairs[k].second--;
        
        for (size_t j = i+1; j < Nclauses; j++)
        {
          //skip already considered clauses
          if(count3(j) == 0 || track_clauses(j) == 0)
            continue;
          
          v01 = Vector2i(Midx2(j,0), Midx2(j,1)); sort(v01.begin(), v01.end());
          v02 = Vector2i(Midx2(j,0), Midx2(j,2)); sort(v02.begin(), v02.end());
          v12 = Vector2i(Midx2(j,1), Midx2(j,2)); sort(v12.begin(), v12.end());

          if      (v01 == vpairs[k].first) {
            for (size_t l = 0; l < vpairs.size(); l++)
              if(v02 == vpairs[l].first || v12 == vpairs[l].first){
                vpairs[l].second--;
              }
            
            vpairs[k].second--;
            track_clauses(j) = 0;
          }
          else if (v02 == vpairs[k].first) {
            for (size_t l = 0; l < vpairs.size(); l++)
              if(v01 == vpairs[l].first || v12 == vpairs[l].first){
                vpairs[l].second--;
              }
            
            vpairs[k].second--;
            track_clauses(j) = 0;
          }
          else if (v12 == vpairs[k].first) {
            for (size_t l = 0; l < vpairs.size(); l++)
              if(v01 == vpairs[l].first || v02 == vpairs[l].first){
                vpairs[l].second--;
              }
            
            vpairs[k].second--;
            track_clauses(j) = 0;
          }
        }
        
        vpairsW.push_back(vpairs[k].first);
        if(vpairs[k].second != 0)
          throw runtime_error("Error!");
        
        stable_sort(vpairs.begin(), vpairs.end(),
          [](const pair<Vector2i, int> &a, const pair<Vector2i, int> &b){
            return a.second > b.second;
          }
        );
        
        break;
      }
    }
  }
  idx N = Nvars + vpairsW.size();

  W = MatrixXi::Zero(N, N);
  W.topLeftCorner(Nvars, Nvars) = W1;

  B = VectorXi::Zero(N);
  B.head(Nvars) = B1;
  

  MatrixXi posnegcount = MatrixXi::Zero(vpairsW.size(), 3);

  int sum = 0;
  for (size_t i = 0; i < Nclauses; i++)
  {
    
    Vector2i v01 {Midx(i,0), Midx(i,1)}; sort(v01.begin(), v01.end());
    Vector2i v02 {Midx(i,0), Midx(i,2)}; sort(v02.begin(), v02.end());
    Vector2i v12 {Midx(i,1), Midx(i,2)}; sort(v12.begin(), v12.end());
    
    bool found_pair = false;
    for (size_t k = 0; k < vpairsW.size(); k++)
    {
      if      (v01 == vpairsW[k])
      {
        if (M(i,0)*M(i,1)*M(i,2) > 0)
        {
          W(Midx(i,2), Nvars+k) += 1;
        }else{
          W(Midx(i,2), Nvars+k) -= 1;
        }
        sum ++;
        found_pair = true;
      }
      else if (v02 == vpairsW[k])
      {
        if (M(i,0)*M(i,1)*M(i,2) > 0)
        {
          W(Midx(i,1), Nvars+k) += 1;
        }else{
          W(Midx(i,1), Nvars+k) -= 1;
        }
        sum ++;
        found_pair = true;
      }
      else if (v12 == vpairsW[k])
      {
        if (M(i,0)*M(i,1)*M(i,2) > 0)
        {
          W(Midx(i,0), Nvars+k) += 1;
        }else{
          W(Midx(i,0), Nvars+k) -= 1;
        }
        sum ++;
        found_pair = true;
      }
      
      if(found_pair)
      {
        if(!KZFD)
        {
          if(M(i,0)*M(i,1)*M(i,2) > 0){
              posnegcount(k, 0) += 1;
          }else{
              posnegcount(k, 1) += 1;
          }

          if(posnegcount(k, 2) < posnegcount.row(k).maxCoeff()){
              W(vpairsW[k](0), vpairsW[k](1)) += 1;
              W(vpairsW[k](0), Nvars + k) -= 2;
              W(vpairsW[k](1), Nvars + k) -= 2;
              B(Nvars + k) += 3;
              
              posnegcount(k, 2) += 1;
          }
          
        }else{
          W(vpairsW[k](0), vpairsW[k](1)) += 1;
          W(vpairsW[k](0), Nvars + k) -= 1;
          W(vpairsW[k](1), Nvars + k) -= 1; 
          B(Nvars + k) += 1;
          
          if(M(i,0)*M(i,1)*M(i,2) < 0){
              B(Nvars + k) += 1;
              W(vpairsW[k](0), vpairsW[k](1)) -= 1;
          }
        }

        break;
      }
    }
  }
  // cout << "sum " << sum << endl;
  W += W.transpose().eval();
  return -M;
}

inline MatrixXi cnf_to_PUBO_sparse(
    vector<SMatrixi> &S,
    MatrixXi &W,
    VectorXi &B,
    int &C,
    string path
  )
{
  MatrixXi M;
  idx Nvars = read_cnf(M, path);
  idx Nclauses = M.rows();

  idx N = Nvars;

  vector<vector<Ti>> St(N); // triplet list for sparse init

  W = MatrixXi::Zero(N, N);
  B = VectorXi::Zero(N);
  C = 0;


  MatrixXi Midx = M.cwiseAbs() - MatrixXi::Constant(M.rows(), M.cols(), 1);
  M *= -1;

  for (size_t i = 0; i < Nclauses; i++)
  {
    if(M(i, 2) == 0)
    {  
      if(M(i, 1) == 0)
      { 
        if (M(i,0) > 0) 
        {
          B(Midx(i,0)) += 1;

        }else{
          B(Midx(i,0)) -= 1;
          C += 1;
        }
      }else{
        throw runtime_error("2SAT support!");
      }
      continue;
    }
    if (M(i,0) > 0) {
      if (M(i, 1) > 0) {
        if (M(i, 2) > 0) {
          St[Midx(i,0)].push_back(Ti(Midx(i,1), Midx(i,2), 1));

        }else{
          St[Midx(i,0)].push_back(Ti(Midx(i,1), Midx(i,2), -1));
          W(Midx(i,0), Midx(i,1)) += 1;

        }
      } else {
        if (M(i, 2) > 0) {
          St[Midx(i,0)].push_back(Ti(Midx(i,1), Midx(i,2), -1));
          W(Midx(i,0), Midx(i,2)) += 1;

        }else{
          St[Midx(i,0)].push_back(Ti(Midx(i,1), Midx(i,2), 1));
          W(Midx(i,0), Midx(i,1)) += -1;
          W(Midx(i,0), Midx(i,2)) += -1;
          B(Midx(i,0)) += 1;

        }
      }
    }else{
      if (M(i, 1) > 0) {
        if (M(i, 2) > 0) {
          St[Midx(i,0)].push_back(Ti(Midx(i,1), Midx(i,2), -1));
          W(Midx(i,1), Midx(i,2)) += 1;

        }else{
          St[Midx(i,0)].push_back(Ti(Midx(i,1), Midx(i,2), 1));
          W(Midx(i,0), Midx(i,1)) += -1;
          W(Midx(i,1), Midx(i,2)) += -1;
          B(Midx(i,1)) += 1;

        }
      } else {
        if (M(i, 2) > 0) {
          St[Midx(i,0)].push_back(Ti(Midx(i,1), Midx(i,2), 1));
          W(Midx(i,0), Midx(i,2)) += -1;
          W(Midx(i,1), Midx(i,2)) += -1;
          B(Midx(i,2)) += 1;

        }else{
          St[Midx(i,0)].push_back(Ti(Midx(i,1), Midx(i,2), -1));
          W(Midx(i,0), Midx(i,1)) += 1;
          W(Midx(i,0), Midx(i,2)) += 1;
          W(Midx(i,1), Midx(i,2)) += 1;
          B(Midx(i,0)) += -1;
          B(Midx(i,1)) += -1;
          B(Midx(i,2)) += -1;

          C += 1;
        }
      }
    }
  }

  vector<vector<Ti>> St_sym(N);
  for (size_t i = 0; i < N; i++)
  {
    Vector3i tmp;
    for (auto s: St[i]){
      tmp << i, s.row(), s.col();
      sort(tmp.begin(), tmp.end());

      St_sym[tmp[0]].push_back(Ti(tmp[1], tmp[2], s.value()));
      St_sym[tmp[0]].push_back(Ti(tmp[2], tmp[1], s.value()));
      St_sym[tmp[1]].push_back(Ti(tmp[0], tmp[2], s.value()));
      St_sym[tmp[1]].push_back(Ti(tmp[2], tmp[0], s.value()));
      St_sym[tmp[2]].push_back(Ti(tmp[0], tmp[1], s.value()));
      St_sym[tmp[2]].push_back(Ti(tmp[1], tmp[0], s.value()));
    }
  }
  

  S = vector<SMatrixi>(N, SMatrixi(N, N));
  for (size_t i = 0; i < N; i++)
  {
    S[i].setFromTriplets(St_sym[i].begin(), St_sym[i].end());
    S[i] = S[i].pruned();
  }

  W += W.transpose().eval();

  return -M;
}