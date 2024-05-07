#include "config.hpp"
#include "satreader.hpp"

#include "dg.hpp"

int main(int c_args, char *arg[]) 
{
  json dg_pars = json::parse(ifstream(pth("configs")/"config.json"));

  string output_path = dg_pars["output_path"];

  if(!filesystem::exists(pth(output_path)))
    filesystem::create_directory(pth(output_path));
  

  int max_gwl_steps = dg_pars["max_gwl_steps"];
  int seed = dg_pars["seed"];

  string problem_class = dg_pars["problem_class"];

  if (c_args != 4)
  {
      std::cerr << "Usage: " << arg[0] << " instance_number mappingID runID\n";
      exit(EXIT_FAILURE);
  }

  string runID(arg[3]);

  int instance = atoi(arg[1]);
  runID += "_" + to_string(instance);

  int mappingID = atoi(arg[2]);
  
  pth save_name("map" + to_string(mappingID) + "_" + runID + ".dat");
  pth barriers_file_name      = pth("barriers_") += save_name;
  pth clusters_file_name      = pth("clusters_") += save_name;
  // pth unstable_states         = pth("unstable_clusters_") += save_name;
  pth log_file_name           = pth("log_") += save_name;
  pth hist_file_name          = pth("hist_") += save_name;

  
  if(!filesystem::exists(pth(output_path)/"logs"))
    filesystem::create_directory(pth(output_path)/"logs");
  #ifndef _WRITE_COUT_
    ofstream logout(pth(output_path) / "logs" / log_file_name);
  #else
    ostream& logout = cout;
  #endif

  logout << "Mapping ID: " << mappingID << endl;
  logout << "Run ID: "     << runID << endl;

  MatrixXi W;
  VectorXi B;
  int C;

  vector<SMatrixi> S0;
  MatrixXi W0;
  VectorXi B0;
  int C0 = 0;

  MatrixXi M;

  string instance_file_name = string(problem_class) + to_string(_N0_) + "-0" + to_string(instance) + ".cnf";
  pth full_path = pth("problems") / problem_class / instance_file_name;

  logout << "Getting mapping of " << full_path << endl;

  if (mappingID == static_cast<int>(MapIDs::PUBO)) {
    M = cnf_to_PUBO_sparse(S0, W0, B0, C0, full_path);

  } else if (mappingID == static_cast<int>(MapIDs::Rosenberg_advanced)) {
    M = QUBO_advanced(W, B, C, full_path, false);
    cnf_to_PUBO_sparse(S0, W0, B0, C0, full_path);

  } else if (mappingID == static_cast<int>(MapIDs::KZFD_advanced)) {
    M = QUBO_advanced(W, B, C, full_path, true);
    cnf_to_PUBO_sparse(S0, W0, B0, C0, full_path);

  } else {
    throw(runtime_error("Wrong mapping id!"));
  }
  
  
  UboLandscape QL(M, _N0_, W, B, C, S0, W0, B0, C0);
  
  const gsl_rng_type *T = gsl_rng_default;
  gsl_rng *r = gsl_rng_alloc(T); gsl_rng_set(r, seed+1);

  auto gen2 = [&r](){return gsl_rng_uniform_int(r, 2);};

  int N = B.size();

  logout << "Native (PUBO) problem size: " << _N0_ << endl;
  if(N != 0) logout << "QUBO mapping size: " << N << endl;


  // Estimate mean random energy (mean at beta = 0)
  VectorXb Xtest(_N0_);
  int Etest = 0;
  for (size_t i = 0; i < 500; i++)
  {
    generate(begin(Xtest), end(Xtest), gen2);
    Etest += QL.Evalue_native(Xtest);
  }
  double mean_E = 1.0*Etest/500;
  logout << "Mean random pubo energy: " << mean_E << endl;


  // Estimate mean random beta
  
  int dEtest = 0;
  for (size_t i = 0; i < 500; i++)
  {
    int dEtmp = 0;

    while(dEtmp == 0)
    {
      generate(begin(Xtest), end(Xtest), gen2);
      int j = gsl_rng_uniform_int(r, _N0_);
      dEtmp = abs(QL.dE_native(Xtest, j));
    }
    dEtest += dEtmp;
  }

  logout << "Mean random dE_native: " << 1.*dEtest/500 << endl;
  double beta = 500.0/dEtest;
  
  //Number of steps in a single basin limit
  int restart_limit = _RESTART_LIMIT_;

  //Number of steps at high energies limit
  int high_e_limit = _HIGH_E_LIMIT_;
  
  //multiplier to adjust beta
  int beta_scale = dg_pars["beta_scale"];
  beta *= beta_scale;
  logout << "Beta = 1/T = beta_scale/<dE> " << beta << endl;

  //multiplier to adjust upper energy of GWL
  double E_scale = dg_pars["E_scale"];
  VectorXd E_partition = VectorXd::LinSpaced((int)(E_scale*mean_E) + 1, 0, (int)(E_scale*mean_E));
  
  logout << "Energy partition size: " << E_partition.size() << endl;
  logout << "Energy partition" << endl << E_partition.transpose() << endl;
  logout << "Energy partition max E (<E> x E_scale)" << endl << E_partition.transpose() << endl;

  // Use steepest descent if needed
  QL.is_steepest = false;

  //maximum energy at which BFS is performed
  int BFS_E = dg_pars["BFS_E_max"]; 

  logout << "QUBO mapping landscape: " << QL.qubo_landscape << endl;
  if(QL.qubo_landscape)
    logout << "QUBO barrier factor: " << _QUBO_BAR_FACTOR_ << endl;
  
  int K = dg_pars["K_basins"];

  logout << "Basin partition size: " << K << endl;
  #ifdef _BALLISTIC_SEARCH_
  logout << "Local search heuristic: " << "Ballistic search" << endl;
  #else
  logout << "Local search heuristic: " << "Random search" << endl;
  logout << "Random search limit: " << _RS_LIMIT_ << endl;
  #endif

  logout << "GWL BFS LIMIT " << _GWL_BFS_LIMIT_ << endl;
  logout << "CUMULATIVE BFS LIMIT " << _CUMULATIVE_BFS_LIMIT_ << endl;

  pth histogram_path = pth(output_path) / "histograms";
  if(!filesystem::exists(pth(output_path) / "histograms"))
    filesystem::create_directory(pth(output_path) / "histograms");
  pth barrier_path = pth(output_path) / "barriers";
  if(!filesystem::exists(pth(output_path) / "barriers"))
    filesystem::create_directory(pth(output_path) / "barriers");
  pth cluster_path = pth(output_path) / "clusters";
  if(!filesystem::exists(pth(output_path) / "clusters"))
    filesystem::create_directory(pth(output_path) / "clusters");

  //maximum threads to use for parallel sampling
  int MAX_TH = 1;

// Calculating minima
#ifdef _OPENMP
  MAX_TH = omp_get_max_threads();
  if (dg_pars["num_threads"] < MAX_TH){
    MAX_TH = dg_pars["num_threads"];
  }
#else 
  logout << "OMP not included" << endl;
#endif

  //MAX_TH parallel samples of states in basins and barriers
  vector<pair<ordered_setvector<_N0_>, MatrixXi>> basins_and_barriers(MAX_TH); 

  logout << "OMP using threads: " << MAX_TH << endl;
  logout << "Max sweeps per thread: " << max_gwl_steps << endl;

#ifdef _OPENMP 
  #pragma omp parallel for
  for (size_t th = 0; th < MAX_TH; th++) 
  { 
    ostream* threadlog = NULL;
    if(th == 0){
      threadlog = &logout;
    }else{
      threadlog = new ofstream(pth(output_path) / pth("logs") / (pth(to_string(th)+"th_") += log_file_name));
    }
    
    bool run_final_bfs = false;
    if(th == 0)
      run_final_bfs = true; // only one thread needs to run breadth first search
    
    basins_and_barriers[th] = QL.run_GWLBS(beta, 
                              E_partition, 
                              K, 
                              max_gwl_steps,
                              restart_limit,
                              high_e_limit,
                              seed+th, 
                              BFS_E,
                              run_final_bfs,
                              (histogram_path / (pth(to_string(th)+"th_") += hist_file_name)).string(),
                              threadlog);

    if(th != 0)
      delete threadlog;
  }

#else

  basins_and_barriers[0] = QL.run_GWLBS(beta, 
                          E_partition, 
                          K, 
                          max_gwl_steps,
                          restart_limit,
                          high_e_limit,
                          seed,
                          BFS_E,
                          true,
                          (histogram_path/hist_file_name).string(),
                          &logout);

#endif
  
  RowVectorXi disconnected_lm = RowVectorXi::Zero(MAX_TH);
  for (size_t th = 0; th < MAX_TH; th++)
  {
    for (size_t i = 0; i < K; i++)
    {
      //Identify disconnected clusters (without found barriers to other clsuters)
      if(-basins_and_barriers[th].second.row(i).sum() + basins_and_barriers[th].second(i,i) == (K-1))
        disconnected_lm[th]++;
    }
  }
  
  logout << "Disconnected clusters" << endl;
  logout << "per thread: " <<  disconnected_lm << endl;
    
  //Postprocessing of sampled states
  vector<vector<pair<int, int>>> states_per_energy(MAX_TH);
  RowVectorXi thread_states = RowVectorXi::Zero(MAX_TH);

  for (size_t th = 0; th < MAX_TH; th++)
  {
    for(auto &basin: basins_and_barriers[th].first)
    {
      thread_states[th] += basin.second.size();
      
      if(states_per_energy[th].size() == 0)
      {
        states_per_energy[th].push_back({basin.first, basin.second.size()});
        
      }else if(states_per_energy[th].back().first == basin.first)
      {
        states_per_energy[th].back().second += basin.second.size();
        
      }else{
        states_per_energy[th].push_back({basin.first, basin.second.size()});
      }
    }
  }
    
  logout << "Per thread found states per energy (including unstable): " << endl;
  for (size_t th = 0; th < MAX_TH; th++)
  {
    MatrixXi matout(2, states_per_energy[th].size());
    for (int i = 0; i < matout.cols(); i++) {
      matout(0, i)  = states_per_energy[th][i].first;
      matout(1, i)  = states_per_energy[th][i].second;
    }
    logout << "thread " << th << ":" << endl << matout << endl;
  }
    
  logout << "Per thread total tracked states (including unstable): " << endl;
  logout << thread_states << " | sum : " << thread_states.sum() << endl;

  logout << "Merging minima and barriers from parallel threads..." << endl;
  
  vector<int> unique_energies;
  
  vector<vector<pair<int, int>>> thUnion;

  vector<VectorXi> v_unique_ids(MAX_TH);
  for (size_t th = 0; th < MAX_TH; th++)
  {
    v_unique_ids[th] = VectorXi::Ones(K);
  }

  vector<vector<MatrixXi>> cmp_pairs(MAX_TH);
  for (size_t th = 0; th < MAX_TH; th++)
  {
    cmp_pairs[th] = vector<MatrixXi>(MAX_TH);

    for (size_t th2 = 0; th2 < MAX_TH; th2++)
    {
      cmp_pairs[th][th2] = MatrixXi::Ones(K, K);
    }
  }
  int cli = 0;
  
newclusterloop:
  for (size_t th = 0; th < MAX_TH; th++) 
  {
    for (size_t th2 = th; th2 < MAX_TH; th2++) 
    {
      auto a = basins_and_barriers[th].first.begin();
      for (int k = 0; k < K; k++) 
      {
        auto b = basins_and_barriers[th2].first.begin();
        for (int j = 0; j < K; j++) 
        {
          if (v_unique_ids[th](k)==1 && v_unique_ids[th2](j)==1 && cmp_pairs[th][th2](k, j)==1)
          {
            if(k != j || th != th2)
            {
              if(a->first == b->first){
                if(th != th2){
                  cmp_pairs[th][th2](k, j) = 0;
                  cmp_pairs[th2][th](j, k) = 0;
                }else{
                  cmp_pairs[th][th2](k, j) =
                  cmp_pairs[th][th2](j, k) = 0;
                }
                
                if (have_common_element(a->second.begin(), a->second.end(), 
                                        b->second.begin(), b->second.end())) 
                {

                  thUnion.push_back(vector<pair<int, int>>());
                  thUnion.back().push_back(pair<int, int>(k, th));
                  thUnion.back().push_back(pair<int, int>(j, th2));

                  unique_energies.push_back(a->first);

                  v_unique_ids[th](k) = v_unique_ids[th2](j) = 0;

                  goto endclusterloop;
                }
              }
            }
          }
          b++;
        }
        a++;
      }
    }
  }
goto endclustercollection;
endclusterloop:

  cli = thUnion.size()-1;
  for (int th = 0; th < MAX_TH; th++) 
  {
    auto a = basins_and_barriers[th].first.begin();
    for (int k = 0; k < K; k++) 
    {
      if (v_unique_ids[th](k)==1) 
      {
        if(a->first == unique_energies[cli])
        {
          bool found_in_cluster = false;
          for(auto &cl : thUnion[cli])
          {
            if(cmp_pairs[th][cl.second](k, cl.first) == 1)
            {
              if(th != cl.second){
                cmp_pairs[th][cl.second](k, cl.first) = 0;
                cmp_pairs[cl.second][th](cl.first, k) = 0;
              }else{
                cmp_pairs[th][cl.second](k, cl.first) = 
                cmp_pairs[th][cl.second](cl.first, k) = 0;
              }
              
              auto &c = basins_and_barriers[cl.second].first[cl.first];
              if(have_common_element(c.second.begin(), c.second.end(), 
                                     a->second.begin(), a->second.end()))
              {
                
                thUnion[cli].push_back(pair<int, int>(k, th));
                v_unique_ids[th](k) = 0;

                found_in_cluster = true;
                th = -1;
                break;
              }
            }
          }
          if(found_in_cluster)
            break;
        }
      }
      a++;
      if(k+1 == K && th+1 == MAX_TH){
        goto newclusterloop;
      }
    }
  }          
endclustercollection:

  for(auto& t: thUnion){
    auto cl = t[0];
    auto &c0 = basins_and_barriers[cl.second].first[cl.first];

    for (size_t i = 1; i < t.size(); i++)
    {
      cl = t[i];
      auto &c1 = basins_and_barriers[cl.second].first[cl.first];
      c0.second.merge(c1.second);
    }
  }

  logout << "Collecting complete!" << endl;
  int thUsize = thUnion.size();

  int total_count = 0;
  vector<pair<int, int>> unq_clust_idx_th;

  for(int th = 0; th < MAX_TH; th++)
  {
    total_count += v_unique_ids[th].sum();

    for(int i = 0; i < v_unique_ids[th].size(); i++)
    {
      if(v_unique_ids[th][i] != 0)
      {
        unq_clust_idx_th.push_back({i, th});
        unique_energies.push_back(basins_and_barriers[th].first[i].first);
      }
    }
  }

  total_count += thUsize;

  logout << "Threads common clusters between threads: " << thUsize << endl;
  logout << "Threads unique clusters in all threads: " << total_count - thUsize << endl;

  logout << "Total clusters found: " << total_count << endl;

  RowVectorXi cluster_sizes = RowVectorXi::Zero(total_count);
  for (size_t i = 0; i < thUsize; i++)
  {
    auto cl = thUnion[i][0];
    auto &c = basins_and_barriers[cl.second].first[cl.first];

    cluster_sizes[i] = c.second.size();
  }

  cluster_sizes(seq(0, cluster_sizes.size()-1)) = Map<RowVectorXi>(cluster_sizes.data(), cluster_sizes.size());

  MatrixXi UB = MatrixXi::Constant(total_count, total_count, -1); // UB = unique barriers

  for (int i = 0; i < thUsize; i++) 
  {
    for (int j = i+1; j < thUsize; j++)
    {
      for(auto &cl1: thUnion[i]){
        int th1 = cl1.second;
        for(auto &cl2: thUnion[j]){
          int th2 = cl2.second;
          if(th1 == th2)
          {
            int tmp_bar = basins_and_barriers[th1].second(cl1.first, cl2.first);

            if(tmp_bar != -1){
              if(tmp_bar < UB(i, j) || UB(i, j) == -1)
              {
                UB(i, j) = UB(j, i) = tmp_bar;
              }
            }
          }
        }
      }
    } 
    for (size_t unqi = thUsize; unqi < unq_clust_idx_th.size() + thUsize; unqi++)
    {
      for(auto &cl: thUnion[i])
      {
        int th1 = cl.second; 
        int th2 = unq_clust_idx_th[unqi - thUsize].second;
        int j = unq_clust_idx_th[unqi - thUsize].first;
        if(th1 == th2)
        {
          int tmp_bar = basins_and_barriers[th1].second(cl.first, j);

          if(tmp_bar != -1){
            if(tmp_bar < UB(i, unqi) || UB(i, unqi) == -1)
            {
              UB(i, unqi) = UB(unqi, i) = tmp_bar;
            }
          }
        }
      }
    }
  }

  for (size_t unqi = thUsize; unqi < unq_clust_idx_th.size() + thUsize; unqi++)
  {
    int i = unq_clust_idx_th[unqi - thUsize].first;
    int th1 = unq_clust_idx_th[unqi - thUsize].second;
    for (size_t unqj = unqi+1; unqj < unq_clust_idx_th.size() + thUsize; unqj++)
    {
      int j = unq_clust_idx_th[unqj - thUsize].first;
      int th2 = unq_clust_idx_th[unqj - thUsize].second;
      if(th1 == th2)
      {
        int tmp_bar = basins_and_barriers[th1].second(i, j);

        if(tmp_bar != -1){
          if(tmp_bar < UB(unqi, unqj) || UB(unqi, unqj) == -1)
          {
            UB(unqi, unqj) = UB(unqj, unqi) = tmp_bar;
          }
        }
      }
    }

    cluster_sizes(unqi) = basins_and_barriers[th1].first[i].second.size();
  }

  if(unique_energies.size() != UB.rows()){
    throw runtime_error("Unique barriers error!");
  }

  vector<bool> saddle_states(UB.rows(), false);
  for(int i = 0; i < UB.rows(); i++){
    UB(i, i) = unique_energies[i];

    for(int j = 0; j < UB.rows(); j++){
      if(i != j && UB(i, i) == UB(i, j))
      {
        saddle_states[i] = true;
        break;
      }
    }
  }

  bool zero_barrier = false;
  for(int i = 0; i < UB.rows(); i++){
    for(int j = i+1; j < UB.cols(); j++){
      if(UB(i, j) == UB(i, i) &&
         UB(i, j) == UB(j, j))
      {
        logout << i << " | " << j << endl;
        zero_barrier = true;
      }

      if(UB(i, j)!= -1 && (UB(i, j) < UB(i, i) || UB(i, j) < UB(j, j)))
      {
        cout << endl << UB(i, j) << " " << UB(i, i) << " " << UB(j, j) << endl;
        logout << "^^ Negative Barrier! ^^" << endl;
      }
    }
  }

  if(zero_barrier)
    logout << "^^ Warning: ZERO BARRIERS MET ^^" << endl; 
  
  vector<int> disconnected_states;
  for (size_t i = 0; i < UB.rows(); i++)
  {
    if(-UB.row(i).sum() + UB(i,i) == (UB.rows()-1))
    {
      disconnected_states.push_back(i);
    }
  }
  
  vector<pair<int, int>> final_states_per_energy;
  
  for(int i = 0; i < UB.rows(); i++)
  {
    int Etmp = UB(i, i);
    auto a = find_if(final_states_per_energy.begin(), final_states_per_energy.end(),
                     [Etmp](const pair<int, int>& p) { return p.first == Etmp; });
    
    if(a == end(final_states_per_energy)){
      final_states_per_energy.push_back({UB(i, i), cluster_sizes(i)});
      
    }else
    {
      a->second += cluster_sizes(i);
    }
  }
  sort(final_states_per_energy.begin(), final_states_per_energy.end(), 
       [](const pair<int, int>& p1, const pair<int, int>& p2) { return p1.first < p2.first; });
  
  
  logout << "Final disconnected clusters: " << disconnected_states.size()  << endl;
  logout << "Final total tracked states (including unstable):  " << endl;

  MatrixXi matout(2, final_states_per_energy.size());
  for (int i = 0; i < matout.cols(); i++) {
    matout(0, i)  = final_states_per_energy[i].first;
    matout(1, i)  = final_states_per_energy[i].second;
  }
  logout << matout << endl;
  
  logout << "Final total states: " << cluster_sizes.sum() << endl;

  ofstream fout(pth(barrier_path) / barriers_file_name);
  fout << UB << endl;
  fout.close();

  // fout.open(pth(cluster_path) / unstable_states);
  // fout << cluster_sizes << endl;
  // fout.close();
  
  vector<pair<int, set<bitvec<_N0_>, VecCmp<_N0_>>>> output_clusters;
  RowVectorXi cluster_sizes_stable = cluster_sizes;

  int all_states_size = cluster_sizes.sum();
  logout << "Removing unstable states... " << endl;
  
  for (size_t i = 0; i < thUsize; i++)
  {
    auto cl = thUnion[i][0];
    auto &c = basins_and_barriers[cl.second].first[cl.first];
    
    output_clusters.push_back(move(c));
   
    for(auto v = output_clusters.back().second.begin(); v!= output_clusters.back().second.end();)
    {
      size_t a;
      for (a = 0; a < _N0_; a++)
      {
        if(!QL.qubo_landscape){
          if(QL.dE_native(toVec<_N0_>(*v), a) < 0)
          {
            cluster_sizes_stable(i)--;
            
            v = output_clusters.back().second.erase(v);

            saddle_states[i] = true;
            break;
          }
          
        }else{
          if(QL.dE_native(toVec<_N0_>(*v), a) < 0 && QL.dEqubo(toVec<_N0_>(*v), a, _QUBO_BAR_FACTOR_, -1) < 0)
          {
            cluster_sizes_stable(i)--;

            v = output_clusters.back().second.erase(v); 
            
            saddle_states[i] = true;
            break;
          }
        }
      }
      if(a == _N0_)
        v++;
    }
    
    if(cluster_sizes_stable(i) == 0)
      logout << "Warning: empty cluster!" << endl;
  }
  
  for (size_t unqi = thUsize; unqi < unq_clust_idx_th.size() + thUsize; unqi++)
  {
    int i = unq_clust_idx_th[unqi - thUsize].first;
    int th = unq_clust_idx_th[unqi - thUsize].second;

    auto &c = basins_and_barriers[th].first[i];

    output_clusters.push_back(move(c));    

    for(auto v = output_clusters.back().second.begin(); v!= output_clusters.back().second.end();)
    {
      size_t a;
      for (a = 0; a < _N0_; a++)
      {
        if(!QL.qubo_landscape){
          if(QL.dE_native(toVec<_N0_>(*v), a) < 0)
          {
            cluster_sizes_stable(unqi)--;
            
            v = output_clusters.back().second.erase(v);

            saddle_states[unqi] = true;
            break;
          }
          
        }else{
          if(QL.dE_native(toVec<_N0_>(*v), a) < 0 && QL.dEqubo(toVec<_N0_>(*v), a, _QUBO_BAR_FACTOR_, -1) < 0)
          {
            cluster_sizes_stable(unqi)--;
            
            v = output_clusters.back().second.erase(v);
            
            saddle_states[unqi] = true;
            break;
          }
        }
      }
      if(a == _N0_)
        v++;
    }

    if(cluster_sizes_stable(unqi) == 0)
      logout << "Warning: empty cluster!" << endl;
      
  }
  

  final_states_per_energy = vector<pair<int, int>>();

  for(int i = 0; i < UB.rows(); i++)
  {
    int Etmp = UB(i, i);
    auto a = find_if(final_states_per_energy.begin(), final_states_per_energy.end(),
                    [Etmp](const pair<int, int>& p) { return p.first == Etmp; });
    
    if(a == end(final_states_per_energy)){
      final_states_per_energy.push_back({UB(i, i), cluster_sizes_stable(i)});
      
    }else
    {
      a->second += cluster_sizes_stable(i);
    }
  }
  sort(final_states_per_energy.begin(), final_states_per_energy.end(), 
      [](const pair<int, int>& p1, const pair<int, int>& p2) { return p1.first < p2.first; });
  
  logout << "Final STABLE states per energy: " << endl;

  matout = MatrixXi(2, final_states_per_energy.size());
  for (int i = 0; i < matout.cols(); i++) {
    matout(0, i)  = final_states_per_energy[i].first;
    matout(1, i)  = final_states_per_energy[i].second;
  }
  logout << matout << endl;
  
  logout << "Fraction Stable/(Stable+Unstable) " << 100.0*cluster_sizes_stable.sum()/all_states_size << "% states" << endl;
  logout << "Final STABLE states: " << cluster_sizes_stable.sum() << endl;

  fout.open(pth(cluster_path) / clusters_file_name);
  fout << cluster_sizes_stable << endl;
  fout.close();
  

  if(dg_pars["save_local_minima"])
  {
    logout << "Saving perceived local minima..." << endl;

    if(not filesystem::exists(pth(output_path)/"local_minima"))
      filesystem::create_directory(pth(output_path)/"local_minima");
  
    pth lm_path = pth("local_minima")/pth("lm_") += to_string(mappingID) + "_" + runID;

    if(filesystem::exists(pth(output_path)/lm_path)){
      for (const auto &cluster_file : filesystem::directory_iterator(pth(output_path)/lm_path))
      {
        filesystem::remove(cluster_file.path());
      }
    }else{
      filesystem::create_directory(pth(output_path)/lm_path);
    }

    for(int i = 0; i < output_clusters.size(); i++)
    {
      if(saddle_states[i] ||
        !(find(disconnected_states.begin(), disconnected_states.end(), i) == end(disconnected_states)))
      {
        continue;
      }

      fout.open(pth(output_path)/lm_path/pth(to_string(i) + ".dat"));
      
      for(auto &av: output_clusters[i].second)
        fout << toVec<_N0_>(av).transpose() << endl;
      
      fout.close();
    }

    logout << "States saved!" << endl;
  }

  return 0;
}
