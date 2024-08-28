/*-----------------------------------------------------------------------------

Markov Chain Monte Carlo C++ interface declaration

-----------------------------------------------------------------------------*/

int run_energy_sat(
        int* x_start, int x_start_size,
        int* J_indices, int J_indices_size,
        int* J_values, int J_values_size,
        int* h, int h_size);


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
        double initial_T, double final_T,

        char * T_schedule,
        
        int aux_size, int track_stats_freq);
      
