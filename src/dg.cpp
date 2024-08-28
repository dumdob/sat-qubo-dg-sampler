#include "dg.hpp"

UboLandscape::UboLandscape(
    const MatrixXi &M, int N0, 
    const MatrixXi &W, const VectorXi &B, int C,
    const vector<SMatrixi>& S0,
    const MatrixXi &W0, const VectorXi &B0, int C0)
    : M(M), N0(N0), W(W), B(B), C(C), S0(S0), W0(W0), B0(B0), C0(C0) 
{
  N = B.size();

  if(N == 0){
    qubo_landscape = false;
  }else{
    qubo_landscape = true;
  }
}

int UboLandscape::Evalue_native(const VectorXb& Xb)
{
  
  int E = (Xb.cast<int>().transpose() * W0 * Xb.cast<int>() / 2 + Xb.cast<int>().transpose() * B0)(0) + C0;

  if(S0.size() != 0){
    for (size_t i = 0; i < N0; i++)
    {
      for (size_t si = 0; si < S0[i].outerSize(); si++) {
        for (SMatrixi::InnerIterator it(S0[i], si); it; ++it) {
          if(i < it.row() && it.row() < it.col())
          {
            E += it.value()*Xb(i)*Xb(it.row())*Xb(it.col());
          }
        }
      }
    }
  }
  
  return E;
}

int UboLandscape::Evalue(const VectorXb& Xb)
{
  int E = (Xb.cast<int>().transpose() * W * Xb.cast<int>() / 2 + Xb.cast<int>().transpose() * B)(0) + C;
  
  return E;
}

int UboLandscape::dE_native(const VectorXb& Xb, int i)
{
  int df = (B0(i) + (W0.row(i) * Xb.cast<int>())(0));

  if(S0.size() != 0)
  {
    for (size_t si = 0; si < S0[i].outerSize(); si++) {
      for (SMatrixi::InnerIterator it(S0[i], si); it; ++it) {
        if(it.row() < it.col())
        {
          df += it.value()*Xb(it.row())*Xb(it.col());
        }
      }
    }
  }
  
  return df*(-2 * Xb(i) + 1);
}


int UboLandscape::dEqubo(const VectorXb &Xb, int xk, int Factor, int return_type)
{
  vector<int> neutral_y;
  vector<pair<int, int>> F_y;
  
  int df_X = B(xk) + (W(xk, seq(0, N0-1))*Xb.cast<int>())(0); //F=0 gradient
  
  int Fall = 0;
  for (size_t i = N0; i < N; i++)
  {
    if(W(xk, i) != 0) //only if xk interacts with an aux. variable
    {
      Fall++;
      int df = B(i) + (W(i, seq(0, N0-1)) * Xb.cast<int>())(0); //gradient for y_i
      int df2 = Xb(xk) ? df - W(i, xk): df + W(i, xk); //gradient for y_i when x_k is flipped
      
      if(df < 0) //if df < 0, then y_i should be in state 1
      {
        df_X += W(xk, i); //F=0 gradient

        if(df2 > 0)
          F_y.push_back({i, df2});

      }else if(df == 0)
      {
        neutral_y.push_back(i);
        if(df2 != 0)
          F_y.push_back({i, df2});

      }else
      {
        if(df2 < 0)
          F_y.push_back({i, -df2});

      }
      
    }
  }
  stable_sort(F_y.begin(), F_y.end(),
    [](const pair<int, int> &a, const pair<int, int> &b){return abs(a.second) > abs(b.second);});

  int F = Fall > Factor ? Factor: Fall;
  
  int df3 = 0;
  int count_F = 0;
  for (size_t i = 0; i < F_y.size(); i++) //check barriers for neutral_y = 0...0
  {
    if(count_F == F)
      break;
    if(!binary_search(neutral_y.begin(), neutral_y.end(), F_y[i].first))
    {
      df3 += F_y[i].second;
      count_F++;
    }else
    {
      if(F_y[i].second < 0){
        df3 += F_y[i].second;
        count_F++;
      }
    }
  }
  int min_df = df_X*(-2*Xb(xk) + 1) - df3;
  
  if(return_type == -1 && min_df < 0){
    return min_df;
  }else if(return_type == 0 && min_df <= 0){
    return min_df;
  }

  int neutral_y_all = pow(2, neutral_y.size());

  for (size_t i = 1; i < neutral_y_all; i++) //check barriers for all neutral_y
  {
    int df_X2 = df_X;
    vector<int> neutral_y_ones;
    for (size_t sh = 0; sh < neutral_y.size(); sh++){
      if((i >> sh) & 1)
      {
        df_X2 += W(xk, neutral_y[sh]);
        neutral_y_ones.push_back(neutral_y[sh]);
      }
    }

    df3 = count_F = 0;
    for (size_t i = 0; i < F_y.size(); i++)
    {
      if(count_F == F)
        break;
      if(!binary_search(neutral_y.begin(), neutral_y.end(), F_y[i].first))
      {
        df3 += F_y[i].second;
        count_F++;
      }else
      {
        if(F_y[i].second < 0 && 
          !binary_search(neutral_y_ones.begin(), neutral_y_ones.end(), F_y[i].first))
        {
          df3 += -F_y[i].second;
          count_F++;

        }else if(F_y[i].second > 0 && 
          binary_search(neutral_y_ones.begin(), neutral_y_ones.end(), F_y[i].first))
        {
          df3 += F_y[i].second;
          count_F++;
        }
      }
    }

    int df = df_X2*(-2*Xb(xk) + 1) - df3;

    if(df < min_df){
      min_df = df;
    }
    if(return_type == -1 && min_df < 0){
      return min_df;
    }else if(return_type == 0 && min_df <= 0){
      return min_df;
    }
  }

  return min_df;
}

vector<E_set<_N0_>> UboLandscape::random_descent(const VectorXb &X0, int *df_out, int rndseed)
{
  VectorXb X(X0);

  vector<E_set<_N0_>> interim_states;

  const gsl_rng_type *T = gsl_rng_default;
  gsl_rng *r = gsl_rng_alloc(T);
  gsl_rng_set(r, rndseed);

  vector<int> perm(N0);
  std::iota(perm.begin(), perm.end(), 0);

  bool found_descent = false;  
  int random_search_count = 0;

  int df_total = 0;

  int df, dfq;
  int i1, i2, i3;

  vector<int> zero_df_idx;
  vector<pair<int, int>> zero_df_idx_qubo;
  
  while (true)
  {
    gsl_ran_shuffle(r, perm.data(), N0, sizeof(int));

    for (size_t i = 0; i < N0; i++) {
      i2 = perm[i];

      df = dE_native(X, i2);

      if (qubo_landscape){
        if(df < 0)
        {
          dfq = dEqubo(X, i2, _QUBO_BAR_FACTOR_, -1); // finds if dEq < 0, else outputs the minimum >=0 

        }else if (df == 0)
        {
          dfq = dEqubo(X, i2, _QUBO_BAR_FACTOR_, 0); // finds if dEq == 0
        }
      }

      if(df == 0 && (!qubo_landscape || dfq == 0)) // if dEpubo = 0 and dEqubo = 0
        zero_df_idx.push_back(i2);

      if(df < 0  && (!qubo_landscape || dfq == 0)) // if dEpubo < 0 but dEqubo = 0
        zero_df_idx_qubo.push_back({i2, df});
      
      if(df < 0 && (!qubo_landscape || dfq < 0))  // if dEpubo < 0 and dEqubo < 0
      {
        i3 = i2;
        found_descent = true;

        zero_df_idx.clear();
        zero_df_idx_qubo.clear();
        
        break;
      }
    }

    if (found_descent)
    {
      X(i3) = !X(i3);
      df_total += df;
      
      found_descent = false;
      random_search_count = 0;
      
    }else if((zero_df_idx.size() + zero_df_idx_qubo.size()) == 0)
    { 
      if(interim_states.size() > 0 && df_total == interim_states.back().first)
        break;

      interim_states.push_back(E_set<_N0_>(df_total, {tobitvec<_N0_>(X)}));
      
      break; //all gradients positive

    }else if(random_search_count >= _RS_LIMIT_){
      break;

    }else
    {
      if(interim_states.size() == 0 || df_total != interim_states.back().first)
        interim_states.push_back(E_set<_N0_>(df_total, {tobitvec<_N0_>(X)}));
      
      for(auto d: zero_df_idx){
        X(d) = !X(d);
        interim_states.back().second.insert(tobitvec<_N0_>(X));
        X(d) = !X(d);
      }

      i1 = gsl_rng_uniform_int(r, zero_df_idx.size() + zero_df_idx_qubo.size());

      if(i1 < zero_df_idx.size()){
        i3 = zero_df_idx[i1];
        random_search_count++;
        
      }else{
        i3 = zero_df_idx_qubo[i1 - zero_df_idx.size()].first;
        df_total += zero_df_idx_qubo[i1 - zero_df_idx.size()].second;
        random_search_count = 0;

      }
      X(i3) = !X(i3);


      zero_df_idx.clear();
      zero_df_idx_qubo.clear();
      
    }
  }
  
  if (df_out) {
    *df_out = df_total;
  }
  
  gsl_rng_free(r);

  return interim_states;
}

template <typename Scalar>
void insert_cluster(int k, MatrixXi& barriers)
{
  int maxK = barriers.rows();

  if(k > 0){
    barriers(seq(0, k-1), seq(k+1, last)) = barriers(seq(0, k-1), seq(k, last-1)).eval();
    barriers(seq(k+1, last), seq(0, k-1)) = barriers(seq(k, last-1), seq(0, k-1)).eval();
  }

  barriers(seq(k+1, last), seq(k+1, last)) = barriers(seq(k, last-1), seq(k, last-1)).eval();
  barriers(all, k) = VectorXi::Constant(maxK, -1);
  barriers(k, all) = RowVectorXi::Constant(maxK, -1);

}


template <typename Scalar>
void unite_clusters(
  const vector<int> &ucv, 
  MatrixXi &barriers)
{
  int maxK = barriers.rows();
  
  auto ucv_reverse = ucv; reverse(ucv_reverse.begin(), ucv_reverse.end());
  
  int ka = ucv[0];

  for(auto kb: ucv_reverse)
  {
    if(ka < kb){
      for (size_t i = 0; i < maxK; i++)
      {
        if(i != ka && i != kb){
          if(barriers(kb, i) != -1){
            if(barriers(ka, i) == -1 || barriers(kb, i) < barriers(ka, i))
            {
              barriers(ka, i) = barriers(i, ka) = barriers(kb, i);
            } 
          }
        }
      }
    }
  }

  int sh_i = 0;
  int sh_j = 0;
  for (size_t i = 0; i < maxK; i++)
  { 
    if(find(++ucv.begin(), ucv.end(), i) == end(ucv))
    {
      sh_j = 0;
      for (size_t j = 0; j < maxK; j++)
      {
        if(find(++ucv.begin(), ucv.end(), j) == end(ucv))
        {
          barriers(i - sh_i, j - sh_j) = barriers(i, j);
        }else{
          sh_j++;
        }

      }
    }else{
      sh_i++;
    }
  }

  int rm_cl = ucv.size()-1;
  barriers(lastN(rm_cl), all) = MatrixXi::Constant(rm_cl, maxK, -1);
  barriers(all, lastN(rm_cl)) = MatrixXi::Constant(maxK, rm_cl, -1);

}


pair<ordered_setvector<_N0_>, MatrixXi>
UboLandscape::run_GWLBS(double beta, const VectorXd &E_partition,
                    int maxK, 
                    int max_attempts,
                    int rndseed,
                    int BFS_E,
                    bool run_final_bfs,
                    const filesystem::path &out_pth,
                    ostream* logout)
{
  int L = E_partition.size();
  
  map<pair<int, int>, Cluster> clusters_map; // ordered container with degenerate stable clusters sorted by energy
  VectorXi outer_theta = VectorXi::Zero(L);  // the histogram for the states not tracked by the search
  
  MatrixXi barriers = MatrixXi::Constant(maxK, maxK, -1); //maxK x maxK matrix containing all barriers
  
  // set of allstates (saddles/minima) for each energy
  setmap<_N0_> allstates;
  
  // keep track of the number of discovered stable states at every energy
  vector<int> report_clusters;
  
  const gsl_rng_type *T = gsl_rng_default;
  gsl_rng *r = gsl_rng_alloc(T);
  gsl_rng_set(r, rndseed);
  auto gen2 = [&r](){return gsl_rng_uniform_int(r, 2);};

  int att = 0;

  VectorXb X = VectorXi::Zero(N0).cast<bool>();
  
  generate(begin(X), end(X), gen2);
  
  int Eprev = Evalue_native(X);
  int lprev = L - 1;
  while (Eprev < E_partition[lprev]) {
    lprev--;
  } 

  int kprev = maxK;
  int theta_index_prev = -1;

  int K = 0;

  bool heatup = true; //initial steps to initialize max K basins

  int restart = 0;        //to restart if stuck in some local minimum
  int restart_high_E = 0; //to restart from a low energy state
  bool restarted = false; //restart indicator to correctly choose theta_prev

  int restart_count = 0;
  int restart_high_E_count = 0;
  
  int df_max = 0;
  int flip_count = 0;
  int total_flip_attempts = 0;
  
  int clID = 0;
  int kID = 0;

  int Ebar_outer = -1;
  int kprev_save = -1;

  int all_states = 0;

  auto time_start = chrono::system_clock::now();
  
  while (att <= max_attempts) { //limit the number of GWL steps to max_attempts
    int i = gsl_rng_uniform_int(r, N0);

    int df, df_out, dfq;
    VectorXb X2 = X;

    df = dE_native(X, i);
    dfq = qubo_landscape ? dEqubo(X, i, _QUBO_BAR_FACTOR_, 1) : df;
    
    assert(dfq >= df && "Wrong qubo gradient: df (qubo) < df (pubo)!");
    
    if(abs(dfq) > df_max){
      df_max = abs(dfq);
    }
    X2(i) = !X2(i);

    vector<E_set<_N0_>> Xdescent = random_descent(X2, &df_out, rndseed + att);

    if(Xdescent.size() == 0)
      throw runtime_error("Xdescent size = 0!");

    int l = L - 1;
    while (Eprev + df < E_partition[l]) {
      l--;
    }
    
    if(!heatup){
      if(l >= L-1){
        restart_high_E++;
      }
      
      //low T MCMC when too high in energy __MAX_RESTARTS__ times (optional)
      if(restart_high_E == __RESTARTS_PERIOD__ && restart_high_E_count < __MAX_RESTARTS__)
      {
        restart_high_E = 0;
        restart_high_E_count++;

        for (size_t i = 0; i < 10*N0; i++)
        {
          int i1 = gsl_rng_uniform_int(r, N0);;

          for (size_t j = i1; j < i1 + N0; j++)
          {
            int demc = dE_native(X, j%N0);

            if (gsl_rng_uniform(r) < exp(-10*beta*demc))
            {
              X(j%N0) = !X(j%N0);
            }
          
          }
        }

        Eprev = Evalue_native(X);
        lprev = L - 1;
        while (Eprev < E_partition[lprev]) {
          lprev--;
        }
        kprev = maxK;

        theta_index_prev = -1;

        kprev_save = -1;
        Ebar_outer = -1;
        
        restarted = true;
        att--;
        
        continue;
      }
    }
    
    int Etmp = Eprev + df + df_out;
    if(Etmp < 0){
      throw runtime_error("Negative energy!");
    }

    double p;
    K = clusters_map.size() < maxK ? clusters_map.size() : maxK;

    if (K < maxK)
    {
      att--;
      bool lm_exists = false;
      int x0 = Xdescent.size()-1;
      if(!heatup){
        cout << "\r number of basins is < K, looking for more... " << flush;
        x0 = 0; // add, if escaped saddles are to be considered 
      }
      for (size_t x = x0; x < Xdescent.size(); x++)
      {
        int Etmpx = Eprev + df + Xdescent[x].first;

        auto searchE = allstates.find(Etmpx);
        
        if(searchE != end(allstates))
        {
          for(auto &v : Xdescent[x].second)
            //could not fix the compiler error using std::less for heterogeneous lookup
            if (auto search = searchE->second.find({-1, v});
                search != searchE->second.end())
            {
              lm_exists = true;
              break;
            }
        }

        if (!lm_exists && (heatup || Etmpx >= clusters_map.rbegin()->first.first)) {
          clID++; 

          if(searchE != end(allstates))
          {
            // for(auto &v: Xdescent[x].second)
            //   searchE->second.insert({clID, tobitvec<_N0_>(v)});

            //execute breadth-first search to expand the cluster
            run_bfs(Xdescent[x].second, clID, searchE->second);

          }else{
            auto handle = allstates.insert({Etmpx, idset<_N0_>()});
            
            // for(auto &v: Xdescent[x].second)
            //   handle.first->second.insert({clID, tobitvec<_N0_>(v)});

            //execute breadth-first search to expand the cluster
            run_bfs(Xdescent[x].second, clID, handle.first->second);
            
            report_clusters.push_back(handle.first->second.size());
            
          }
          kID++;
          clusters_map.insert(pair{pair<int, int>(Etmpx, kID), 
                                    Cluster(set<int>({clID}), false, VectorXi::Zero(L))});
        }

      }
      
      if (clusters_map.size() >= maxK) {
        if(heatup)
        {
          heatup = false;

          if(logout)
          {
            (*logout) << "Basin pool K: " << clusters_map.size() << endl;
            
            (*logout) << "Basin energies:" << endl;
            for (auto &a : clusters_map) {
              (*logout) << a.first.first << " ";
            }
            (*logout) << endl;
          }
          
        }else{
          cout << " Found K basins" << endl;
        }
        
        kprev_save = -1;
        Ebar_outer = -1;

        kprev = maxK;

        lprev = L-1;
      }

      generate(begin(X), end(X), gen2);
      Eprev = Evalue_native(X);
      
      restarted = true;
      theta_index_prev = -1;

    } else 
    {  
      bool new_insert = false;
      bool clusters_united = false;
      
      int k = 0;
      
      vector<pair<int, int>> zbseq(Xdescent.size(), pair<int, int>(-1, -1)); //zero barrier sequence
      
      vector<vector<int>> ucv_ids(Xdescent.size());      //unite allstates vector of indices
      
      vector<vector<map<pair<int, int>, Cluster>::iterator>>
        ucv_iter(Xdescent.size()); //unite allstates vector of iterators
      
      vector<set<int>> vfoundIDs(Xdescent.size());  //vector of found IDs of states in energy sets
      
      int current_maxE;
      
      if(clusters_map.size() == maxK){
        current_maxE = clusters_map.rbegin()->first.first;
      }else{
        current_maxE = next(clusters_map.begin(), maxK-1)->first.first;
      }
      
      for (size_t x = 0; x < Xdescent.size(); x++)
      {
        Etmp = Eprev + df + Xdescent[x].first;
        
        auto searchE = allstates.find(Etmp);
        
        if(Etmp > current_maxE)
          searchE = end(allstates);
          
        if(searchE != end(allstates)){
          for(auto v = Xdescent[x].second.begin(); v != Xdescent[x].second.end();){
            if (auto search = searchE->second.find({-1, *v}); 
                search != searchE->second.end())
            {
              vfoundIDs[x].insert(search->first);
              v = Xdescent[x].second.erase(v);

            }else{
              v++;
            }
          }
        }
        
        auto a = clusters_map.begin();
        k = 0;
        
        if(Etmp > current_maxE){
          a = end(clusters_map);
          k = maxK;
          
        }else{
          auto search = clusters_map.upper_bound({Etmp, -1}); //find
          if(search == end(clusters_map))
            throw runtime_error("failed to find the energy in clusters_map");
          
          k = distance(clusters_map.begin(), search);
          a = next(a, k);
          while (Etmp == a->first.first && k < K)
          {
            if(vfoundIDs[x].size() > 0 &&
               have_common_element(get<0>(a->second).begin(), get<0>(a->second).end(),
                                   vfoundIDs[x].begin(), vfoundIDs[x].end()))
            {
              ucv_ids[x].push_back(k);
              ucv_iter[x].push_back(a);
            }
            
            a++; k++;
          }
        }
        
        if(k == K)
          k = maxK;
        
        int ucv_size = ucv_ids[x].size();
        if(ucv_size > 0)
        {
          k = ucv_ids[x][0];
          a = ucv_iter[x][0];
          
          get<0>(a->second).merge(vfoundIDs[x]);
          
          for(auto &v: Xdescent[x].second){
            searchE->second.insert({*get<0>(a->second).begin(), v}); //insert vectors in the existing set with existing ID
          }
          
          zbseq[x] = pair<int, int>(k, Etmp);
          
          if(ucv_size > 1)
          {
            for (size_t i = 0; i < x; i++)
            {
              if(zbseq[i].first != -1){
                zbseq[i].first -= (ucv_size-1);
              }
            }
            
            for (size_t i = 0; i < ucv_size-1; i++)
            {
              // merge theta vectors
              auto b = ucv_iter[x][ucv_size-i-1];

              // get<2>(a->second) += get<2>(b->second); //sum of columns
              get<2>(a->second) = get<2>(a->second).cwiseMax(get<2>(b->second)); //max of rows of columns
              
              get<0>(a->second).merge(get<0>(b->second));
              
              if(get<1>(b->second))
                get<1>(a->second) = true;
              
              clusters_map.erase(b);
            }
            
            unite_clusters<bool>(ucv_ids[x], barriers);
            clusters_united = true;
            
          }
        }
        
        if ((x == Xdescent.size()-1) && ucv_size == 0 && Etmp <= current_maxE)
        {
          int clID_insert;
          if(searchE == end(allstates))
          {
            clID++;
            auto handle = allstates.insert({Etmp, idset<_N0_>()});
            
            // for(auto &v: Xdescent[x].second)
            //   handle.first->second.insert({clID, tobitvec<_N0_>(v)});

            run_bfs(Xdescent[x].second, clID, handle.first->second);
            
            report_clusters.push_back(handle.first->second.size());

            clID_insert = clID;

          }else
          {
            if(vfoundIDs[x].size() > 0)
            {
              clID_insert = *vfoundIDs[x].begin();
              
            }else{
              clID++;
              clID_insert = clID;
            }

            // for(auto &v: Xdescent[x].second)
            //   searchE->second.insert({clID_insert, tobitvec<_N0_>(v)});

            run_bfs(Xdescent[x].second, clID_insert, searchE->second);
          }
          
          
          if(Etmp < current_maxE)
          {
            zbseq[x] = pair<int, int>(k, Etmp);
            for (size_t i = 0; i < x; i++)
            {
              if(zbseq[i].first != - 1){
                zbseq[i].first++;
              }
            }
            
            if(K == maxK){
              auto b = next(clusters_map.begin(), K-1);
              get<1>(b->second) = false;
              // outer_theta += get<2>(b->second);
              
              get<2>(b->second) = VectorXi::Zero(L);
            }
            
            kID++;
            clusters_map.insert(pair{pair<int, int>(Etmp, kID),
                                     Cluster(set<int>({clID_insert}), false, VectorXi::Zero(L))});
            
            insert_cluster<bool>(k, barriers);
            
            new_insert = true;
            
          }
        }
        
        if(K > maxK)
          throw runtime_error("Wrong K!");
        
        K = clusters_map.size() < maxK ? clusters_map.size() : maxK;
      }
      
      if(clusters_united)
      {
        int kprev_tmp = kprev;
        if(kprev != maxK)
          for (size_t x = 0; x < ucv_ids.size(); x++){
            for (size_t i = 1; i < ucv_ids[x].size(); i++){
              //adjusting index of previous basin after the unification of allstates
              if(kprev == ucv_ids[x][i]){
                kprev_tmp = ucv_ids[x][0];
                break;
              }
              if(kprev > ucv_ids[x][i] && kprev != maxK){
                kprev_tmp--;
              }
            }
          }
        kprev = kprev_tmp;
        
        int kprev_save_tmp = kprev_save;
        if(kprev_save != -1)
          for (size_t x = 0; x < ucv_ids.size(); x++){
            for (size_t i = 1; i < ucv_ids[x].size(); i++){
              //adjusting index of previous basin after the unification of allstates
              if(kprev_save == ucv_ids[x][i]){
                kprev_save_tmp = ucv_ids[x][0];
                break;
              }
              if(kprev_save > ucv_ids[x][i]){
                kprev_save_tmp--;
              }
            }
          }
        kprev_save = kprev_save_tmp;

        int theta_index_prev_tmp = theta_index_prev;
        if(theta_index_prev != -1 && theta_index_prev != maxK)
          for (size_t x = 0; x < ucv_ids.size(); x++){
            for (size_t i = 1; i < ucv_ids[x].size(); i++){
              //adjusting index of previous basin after the unification of allstates
              if(theta_index_prev == ucv_ids[x][i]){
                theta_index_prev_tmp = ucv_ids[x][0];
                break;
              }
              if(theta_index_prev > ucv_ids[x][i]){
                theta_index_prev_tmp--;
              }
            }
          }
        theta_index_prev = theta_index_prev_tmp;

      }
      
      
      if (new_insert)
      {
        if(kprev >= k && kprev != maxK){  // correct the histogram index after the new basin discovery
          kprev++;
        }

        if(kprev_save >= k && kprev_save != maxK){  // correct the histogram index after the new basin discovery
          kprev_save++;
        }

        if(theta_index_prev >= k && theta_index_prev != maxK){  // correct the histogram index after the new basin discovery
          theta_index_prev++;
        }
      }
      
      if (k == kprev) {
        restart++;
      } else {
        restart = 0;
      }
      
      //adjust the histogram for the zero barriers sequence
      for (size_t i = 0; i < zbseq.size(); i++)
      {
        for (size_t j = i+1; j < zbseq.size(); j++)
        {
          if(zbseq[i].first != -1 && zbseq[j].first != -1)
          {
            int ii = zbseq[i].first;
            int jj = zbseq[j].first;
            
            if(ii != K && jj != K){
              if(ii == jj)
                throw runtime_error("ii == jj");
              
              barriers(ii, jj) = barriers(jj, ii) = zbseq[i].second;
              
              auto b = next(clusters_map.begin(), ii);
              get<1>(b->second) = true;
              
              auto a = next(clusters_map.begin(), jj);
              
              // get<2>(a->second) += get<2>(b->second);
              get<2>(a->second) = get<2>(a->second).cwiseMax(get<2>(b->second));
              get<2>(b->second) = VectorXi::Zero(L);

              if(theta_index_prev == ii){
                theta_index_prev = jj;
              }
            }
            
          }
        }
      }
      map<pair<int, int>, Cluster>::iterator ak, akprev;

      if(k != maxK)
        ak = next(clusters_map.begin(), k);
      
      if(kprev != maxK)
        akprev = next(clusters_map.begin(), kprev);
      
      //modify the barrier matrix
      if(k != kprev && k != maxK && kprev != maxK)
      {
        int Ebar;
        if(dfq > 0){
          Ebar = Eprev + dfq;
        }else{
          Ebar = Eprev;
        }
        
        if(Ebar < barriers(k, kprev) || barriers(k, kprev) == -1)
        {
          barriers(k, kprev) = barriers(kprev, k) = Ebar;
          
          //adjust the histogram if the barrier is zero explicitly
          if(!get<1>(ak->second) && Ebar == ak->first.first)
          {
            get<1>(ak->second) = true;
            // get<2>(akprev->second) += get<2>(ak->second);
            get<2>(akprev->second) = get<2>(akprev->second).cwiseMax(get<2>(ak->second));
            get<2>(ak->second) = VectorXi::Zero(L);

            if(theta_index_prev == k){
              theta_index_prev = kprev;
            }
          }
          
          if(!get<1>(akprev->second) && Ebar == akprev->first.first)
          {
            get<1>(akprev->second) = true;
            // get<2>(ak->second) += get<2>(akprev->second);
            get<2>(ak->second) = get<2>(ak->second).cwiseMax(get<2>(akprev->second));
            get<2>(akprev->second) = VectorXi::Zero(L);
            
            if(theta_index_prev == k){
              theta_index_prev = kprev;
            }
          }
          
          // //indices for the ridge descent
          // int k1, k2;
          // if(kprev < k){
          //   k1 = kprev;
          //   k2 = k;
          // }else{
          //   k1 = k;
          //   k2 = kprev;
          // }

        }
      }

      if(k == maxK && kprev != maxK) //save intermediate energy barrier
      { 
        Ebar_outer = dfq > 0 ? Eprev + dfq: Eprev;
        kprev_save = kprev;
        
      }else if(k != maxK && kprev == maxK && Ebar_outer != -1 && kprev_save != maxK){ 
        //assign saved intermediate energy barrier

        int Ebartmp = dfq > 0 ? Eprev + dfq: Eprev;
        Ebartmp = Ebar_outer > Ebartmp ? Ebar_outer : Ebartmp;

        if(Ebartmp < barriers(k, kprev_save) || barriers(k, kprev_save) == -1){
          barriers(k, kprev_save) = Ebartmp;
          barriers(kprev_save, k) = Ebartmp;
        }
        
      }else if(k != maxK && kprev != maxK){
        kprev_save = -1;
        Ebar_outer = -1;
      }
      
      if(kprev != maxK && Eprev < akprev->first.first)
        throw runtime_error("Error!");
      
      int theta_prev = outer_theta(lprev);
      if (theta_index_prev == -1 || get<1>(next(clusters_map.begin(), theta_index_prev)->second) == true){
        if (theta_index_prev == -1)
          theta_index_prev = kprev;
        
        if(theta_index_prev != maxK)
        {
          auto akth_prev = next(clusters_map.begin(), theta_index_prev);
          
          theta_prev = get<2>(akprev->second)(lprev);
          
          if(auto ittmp = akprev; get<1>(akprev->second)){
            int ktmp = theta_index_prev;
            int lm_E = akth_prev->first.first;
            
            bool local_minimum = false;
            
            while(!local_minimum){
              Vint zerobars;
              int b;
              for(int i = 0; i < barriers.cols(); i++){
                b = barriers(ktmp, i);
                if(b == lm_E){
                  zerobars.push_back(i);
                }
              }
              if(zerobars.size() != 0){
                int ii = gsl_rng_uniform_int(r, zerobars.size());
                // get<2>(next(clusters_map.begin(), zerobars[ii])->second) += get<2>(ittmp->second);
                get<2>(next(clusters_map.begin(), zerobars[ii])->second) 
                  = get<2>(next(clusters_map.begin(), zerobars[ii])->second).cwiseMax(get<2>(ittmp->second));
                get<2>(ittmp->second) = VectorXi::Zero(L);
                
                ktmp = zerobars[ii];
                
                ittmp = next(clusters_map.begin(), ktmp);
                int lm_E2 = ittmp->first.first;
                
                if(lm_E2 >= lm_E)
                {
                  if(logout){
                    (*logout) << "Warning: barrier error kprev: lm_E2 >= lm_E: " << endl;
                    (*logout) << lm_E2 << " " << lm_E << " " << kprev << " " << ktmp << endl;
                  }
                  local_minimum = true;
                }
                
                if(get<1>(ittmp->second))
                {
                  lm_E = lm_E2;
                }else{
                  local_minimum = true;
                }
              }else{
                throw runtime_error("barrier error kprev: zerobars.size() == 0");
              }
            }
            theta_prev = get<2>(ittmp->second)(lprev);
            theta_index_prev = ktmp;
          }
        }
      }else if (theta_index_prev != maxK)
      {
        theta_prev = get<2>(next(clusters_map.begin(), theta_index_prev)->second)(lprev);
      }
      

      int theta_index = k;
      int theta_new = outer_theta(l);
      if(k != maxK){
        theta_new = get<2>(ak->second)(l);
        if(auto ittmp = ak; get<1>(ak->second)){
          int ktmp = k;
          int lm_E = ak->first.first;
          
          bool local_minimum = false;
          
          while(!local_minimum){
            Vint zerobars;
            for(int i = 0; i < barriers.cols(); i++){
              int b = barriers(ktmp, i);
              if(b == lm_E){
                zerobars.push_back(i);
              }
            }
            if(zerobars.size() != 0){
              int ii = gsl_rng_uniform_int(r, zerobars.size());
              
              // get<2>(next(clusters_map.begin(), zerobars[ii])->second) += get<2>(ittmp->second);
              get<2>(next(clusters_map.begin(), zerobars[ii])->second) 
                = get<2>(next(clusters_map.begin(), zerobars[ii])->second).cwiseMax(get<2>(ittmp->second));
              get<2>(ittmp->second) = VectorXi::Zero(L);
              
              ktmp = zerobars[ii];

              ittmp = next(clusters_map.begin(), ktmp);
              int lm_E2 = ittmp->first.first;
              
              if(lm_E2 >= lm_E){
                if(logout){
                  (*logout) << "Warning: barrier error kprev: lm_E2 >= lm_E: " << endl;
                  (*logout) << lm_E2 << " " << lm_E << " " << kprev << " " << ktmp << endl;
                }
                local_minimum = true;
              }
              
              if(get<1>(ittmp->second))
              {
                lm_E = lm_E2;
              }else{
                local_minimum = true;
              }
            }else{
              throw runtime_error("barrier error k: zerobars.size() == 0");
            }
          }
          theta_new = get<2>(ittmp->second)(l);
          theta_index = ktmp;
        }
        
        if (get<1>(next(clusters_map.begin(), theta_index)->second) == true)
          throw runtime_error("theta_index is a saddle cluster!");
        
      }

      if(-beta * df + (theta_prev - theta_new) > 0){
        p = 1;
      }else{
        p = exp(-beta * df + (theta_prev - theta_new));
      }

      bool fliptaken = false;
      total_flip_attempts++;

      int theta_index_prev_report = theta_index_prev;
      if (gsl_rng_uniform(r) < p || restarted)
      {
        flip_count++;
        X(i) = !X(i);
        fliptaken = true;

        Eprev += df;
        
        if (!restarted){
          if(theta_index != maxK){
            get<2>(next(clusters_map.begin(), theta_index)->second)(l)++;
            
          }else{
            outer_theta(l)++;
          }
          
        }else{
          restarted = false;
        }
        
        lprev = l;
        kprev = k;

        theta_index_prev = theta_index;

      } else {

        if(theta_index_prev != maxK){
          get<2>(next(clusters_map.begin(), theta_index_prev)->second)(lprev)++;

        }else{
          outer_theta(lprev)++;
        }

      }
      
      if((att%(max_attempts/10)) == 0)
      {
        all_states = 0;
        int i = 0;
        for(auto &s: allstates)
        {
          report_clusters[i] = s.second.size();
          
          all_states += s.second.size();
          i++;
        }
        
        if(logout){
          vector<VectorXi> satur_tmp(report_clusters.size(), VectorXi::Zero(2));
          for (int i = 0; i < report_clusters.size(); i++)
          {
            satur_tmp[i](0) = next(allstates.begin(), i)->first;
            satur_tmp[i](1) = report_clusters[i];
          }
          sort(satur_tmp.begin(), satur_tmp.end(), 
              [](VectorXi const& t1, VectorXi const& t2){ return t1(0) < t2(0); } );
              
          *logout << endl << "Total states: " << all_states << " at step " << att << "| max tracked energy: " << current_maxE << " | states per energy:" << endl;
          
          MatrixXi saturation_out(2, report_clusters.size());
          for (int i = 0; i < report_clusters.size(); i++)
          {
            saturation_out.col(i) = satur_tmp[i];
          }
          *logout << saturation_out << endl;
        }
        
      }

      //restart when stuck in one basin __MAX_RESTARTS__ times (optional)
      if (restart == __RESTARTS_PERIOD__ && restart_count < __MAX_RESTARTS__) 
      {
        restart_count++;

        restart = 0;
        restarted = true;
        
        generate(begin(X), end(X), gen2);
        kprev = maxK;

        theta_index_prev = -1;

        kprev_save = -1;
        Ebar_outer = -1;

        Eprev = Evalue_native(X);
        lprev = L-1;
        while (Eprev < E_partition[lprev]) {
          lprev--;
        } 
      }

    }

    att++;

  }
  
  K = clusters_map.size() < maxK ? clusters_map.size() : maxK;
  
  auto a = clusters_map.begin();
  for (size_t i = 0; i < K; i++)
  {
    barriers(i, i) = a->first.first;
    a++;
  }

  
  if(logout)
  {
    (*logout) << "Final basin energies:" << endl;
    for (auto a = clusters_map.begin(); a != next(clusters_map.begin(), K); a++) {
      (*logout) << a->first.first << " ";
    }
    (*logout) << endl;

    (*logout) << "Max gradient observed : beta*df:  "
      << beta*df_max << endl;
  
    (*logout) << "Acceptance rate:  " << 1.0*flip_count/total_flip_attempts << endl;
    (*logout) << "Restart count (basin):  " << restart_count << endl;
    (*logout) << "Restart count (high E):  " << restart_high_E_count << endl;
  }

  
  vector<pair<int, set<int>>> Ek_vec(K);
  a = clusters_map.begin(); int k = 0;
  while(k < K){
    Ek_vec[k] = {a->first.first, get<0>(a->second)};
    a++; k++;
  }
  
  all_states = 0;
  int bound_states = 0;
  for(auto &s: allstates)
  {
    if(s.first <= barriers(last, last)){
      bound_states += s.second.size();
    }
    all_states += s.second.size();
  }

  ordered_setvector<_N0_> basins;
  setmap_to_ordered_setvector(allstates, Ek_vec, basins); //allstates is cleared

  int tracked_states = 0;
  int tmpE = basins.begin()->first; //intermediate fix of repeating E_list ids
  for(auto &bas: basins) 
  {
    if(bas.second.size() == 0){
      bas.first = tmpE;
    }else{
      tmpE = bas.first;
      tracked_states += bas.second.size();
    }
  }

  if(logout)
  {
    (*logout) << "Current maximum tracked energy, max E = E(K): " << barriers(last, last) << endl;
    (*logout) << "Tracked states in a thread (k <= K): " << tracked_states << endl;
  }

  auto time_end = chrono::system_clock::now();
  if(logout){
    (*logout) << "(GWL) Sampling time elapsed: " 
      << chrono::duration<double>(time_end - time_start).count() << " (seconds)" << endl;
  }
  
  if(run_final_bfs)
  {
    if(logout)
    {
      (*logout) << "Final BFS for E<=" << BFS_E << " running... " << endl;
      (*logout) << "Per energy limit of states: " << _CUMULATIVE_BFS_LIMIT_ << endl;
    }
    
    int k = 0;
    while(k < basins.size())
    {
      int current_e = basins[k].first;
      if(current_e > BFS_E){
        k++;
        continue;
      }

      int cumulativeCount = basins[k].second.size();
      vector<set<bitvec<_N0_>, VecCmp<_N0_>>*> bas;
      bas.push_back(&basins[k].second);

      k++;

      while(k < basins.size() && basins[k].first == current_e){
        bas.push_back(&basins[k].second);
        cumulativeCount += basins[k].second.size();
        k++;
      }
      int num_bas = bas.size();
      MatrixXi unite_basins_later = MatrixXi::Zero(num_bas, num_bas);

      if(logout)
        (*logout) << "E="<< current_e << ", num of states = " << cumulativeCount << " -> ";

      vector<set<bitvec<_N0_>, VecCmp<_N0_>>> bfs_a(num_bas);
      vector<set<bitvec<_N0_>, VecCmp<_N0_>>> bfs_b(num_bas);

      for (size_t x = 0; x < num_bas; x++)
      {
        for (auto& av: *bas[x])
          bfs_candidates(av, *bas[x], bfs_a[x], bfs_b[x]);
      }

      bool stop_bfs = false;
      while(cumulativeCount < _CUMULATIVE_BFS_LIMIT_ && !stop_bfs)
      {
        stop_bfs = true;
        for (size_t x = 0; x < num_bas; x++)
        {
          if (bfs_b[x].size() != 0){
            stop_bfs = false;
            for(auto &b: bfs_b[x]){
              for (size_t i = 0; i < num_bas; i++)
              {
                if (i != x)
                  if (auto search = bas[i]->find(b); search != bas[i]->end())
                  {
                    if (unite_basins_later(i, x) == 0)
                    {
                      unite_basins_later(i, x) = 1;
                      bas[x]->insert(b);
                    }
                    goto NEXT_b;
                  }
              }
              
              bas[x]->insert(b);
              if (++cumulativeCount >= _CUMULATIVE_BFS_LIMIT_)
                goto ENDBFS;
                
              bfs_candidates(b, *bas[x], bfs_b[x], bfs_a[x]);
                
              NEXT_b:
		            continue;
            } 

            bfs_b[x].clear();
          
          }else if(bfs_a[x].size() != 0){
            stop_bfs = false;
            for(auto &a: bfs_a[x]){
              for (size_t i = 0; i < num_bas; i++)
              {
                if (i != x)
                  if (auto search = bas[i]->find(a); search != bas[i]->end())
                  {
                    if (unite_basins_later(i, x) == 0)
                    {
                      unite_basins_later(i, x) = 1;
                      bas[x]->insert(a);
                    }
                    goto NEXT_a;
                  }
              }

              bas[x]->insert(a);
              if (++cumulativeCount >= _CUMULATIVE_BFS_LIMIT_)
                goto ENDBFS;
            
              bfs_candidates(a, *bas[x], bfs_a[x], bfs_b[x]);
              
              NEXT_a:
		            continue;
            }

            bfs_a[x].clear();
          }
        }
      }

      ENDBFS:
      if(logout){
        if (cumulativeCount >= _CUMULATIVE_BFS_LIMIT_){
          (*logout) << cumulativeCount << " (reached per energy limit)" << endl;
        }else{
          (*logout) << cumulativeCount << endl;
        }
      }
    }
    if(logout)
      (*logout) << "Final BFS complete!" << endl;
  }

  ofstream fout(out_pth);
  MatrixXi theta_out = MatrixXi::Zero(L, maxK+1);
  // MatrixXi theta_out = MatrixXi::Zero(L+1, maxK+1); //for checking local minimum/saddle type

  a = clusters_map.begin();
  for (size_t i = 0; i < K; i++)
  {
    theta_out(seq(0, L-1), i) = get<2>(a->second);
    // theta_out(L, i) = (int)(get<1>(a->second)); //for checking local minimum/saddle type
    
    a++;
  }
  theta_out(seq(0, L-1), maxK) = outer_theta;
  
  //save the histogram
  fout << theta_out << endl;
  fout.close();

  //check the standard deviation of the histogram
  vector<int> theta_nonzero;
  for (auto th: theta_out(seq(0, L-1), all).reshaped()){
    if (th != 0){
      theta_nonzero.push_back(th);
    }
  }
  if (logout){
    (*logout) << "Histogram parameters:" << endl;

    int th_all = std::accumulate(theta_nonzero.begin(), theta_nonzero.end(), 0.0);

    double th_mean = 1.0*th_all/theta_nonzero.size();
    (*logout) << "<th> = " << th_mean << endl;

    double var = 0;
    for(auto th: theta_nonzero)
      var += (th - th_mean) * (th - th_mean);
    
    var /= theta_nonzero.size();
    (*logout) << "sd(th) = " << sqrt(var) << endl;
  }

  gsl_rng_free(r);

  return {basins, barriers};
}

void UboLandscape::run_bfs(const set<bitvec<_N0_>, VecCmp<_N0_>>& states,
                           int clID,
                           idset<_N0_>& allclusters)
{
  int size_count = 0;

  set<bitvec<_N0_>, VecCmp<_N0_>> bfs_a;
  set<bitvec<_N0_>, VecCmp<_N0_>> bfs_b;

  for(auto &av: states)
  {  
    auto insert_test = allclusters.insert({clID, av});
    if(!insert_test.second)
      throw runtime_error("all clusters insertion fail!");

    size_count++;

    bfs_b.insert(av);
  }

  while((bfs_a.size() != 0 || bfs_b.size() != 0) && (size_count < _GWL_BFS_LIMIT_))
  {
    if (bfs_a.size() == 0)
    {
      bfs_candidates(bfs_b, bfs_a, clID, allclusters, size_count);

      bfs_b.clear();
      
    }else
    {
      bfs_candidates(bfs_a, bfs_b, clID, allclusters, size_count);

      bfs_a.clear();
    }
  }

}
void UboLandscape::bfs_candidates(const set<bitvec<_N0_>, VecCmp<_N0_>>& bfs_current, 
                                  set<bitvec<_N0_>, VecCmp<_N0_>>& bfs_new, 
                                  int clID,
                                  idset<_N0_>& bas,
                                  int& size_count)
{
  for(auto& v: bfs_current)
  {
    VectorXb Vb = toVec<_N0_>(v);
    bitvec<_N0_> v2 = v;

    for (size_t i = 0; i < _N0_; i++)
    {
      int df = dE_native(Vb, i);

      if(df == 0 && (!qubo_landscape || dEqubo(Vb, i, _QUBO_BAR_FACTOR_, 0) == 0))
      {
        v2[i] = !v2[i];

        auto insert_test = bas.insert({clID, v2});

        if(insert_test.second){
          size_count++;
          if(size_count >= _GWL_BFS_LIMIT_)
            return;

          bfs_new.insert(v2);
        }

        v2[i] = !v2[i];
      }
    }
  }

}

void UboLandscape::bfs_candidates(const bitvec<_N0_>& v,
                                  const set<bitvec<_N0_>, VecCmp<_N0_>>& bas,
                                  const set<bitvec<_N0_>, VecCmp<_N0_>>& bfs_current,
                                  set<bitvec<_N0_>, VecCmp<_N0_>>& bfs_new)
{
  VectorXb Vb = toVec<_N0_>(v);
  bitvec<_N0_> v2 = v;

  for (size_t i = 0; i < _N0_; i++)
  {
    int df = dE_native(Vb, i);

    if(df == 0 && (!qubo_landscape || dEqubo(Vb, i, _QUBO_BAR_FACTOR_, 0) == 0))
    {
      v2[i] = !v2[i];
     
      if(bas.find(v2) == bas.end() && bfs_current.find(v2) == bfs_current.end())
        bfs_new.insert(v2);

      v2[i] = !v2[i];
    }

  }

}
