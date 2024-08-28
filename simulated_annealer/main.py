import numpy as np
np.seterr(all = "raise")

import time 
import argparse
import sys
import os

from tqdm import tqdm

import concurrent.futures

import json

from typing import Union, List, Tuple, Dict

from pprint import pp
import matplotlib.pyplot as plt

try:
  from c_utils import c_interface as CI
except:
  raise RuntimeError("C++ implementation of MCMC not imported!")

np.random.seed(0)

import logging 
logging.basicConfig(level=logging.WARN)

from scipy.stats import beta

from cnfReader import cnf_reader_to_p
from quboReader import qubo_sat_map

def tts_bootstrap(energy_arrays: List[np.ndarray], 
                  tts_factor: np.ndarray,
                  success_energy_gap: float, 
                  seed = 0,
                  bs_size = 1000,
                  percentiles = [10, 20, 50, 80, 90]) -> Tuple[List[Dict], 
                                                               List[Dict], 
                                                               np.ndarray, 
                                                               np.ndarray]: 
  rs  = np.random.RandomState(seed)

  successes = np.empty(0)
  failures = np.empty(0)

  tts_stats_per_i = []
  suc_stats_per_i = []

  pos_sampled_per_i = []
  for arr in energy_arrays:
    success = (np.array(arr) <= success_energy_gap).sum()
    suc_stats_per_i.append(success)

    fail = len(arr) - success

    successes = np.append(successes, success)
    failures = np.append(failures, fail)

    pos_sampled_per_i.append(beta.rvs(success + 0.5, fail + 0.5, size = bs_size, random_state = rs))

  # Statistics of TTS per each instance
  tts_sampled_per_i = np.log(0.01)/np.log(1.0 - np.clip(np.array(pos_sampled_per_i), 1e-15, 1.0))
  
  for i, tts_i in enumerate(tts_sampled_per_i):
    tts_stats_per_i.append({})

    for ri, r in enumerate(tts_i):
      if r < 1:
        tts_i[ri] = 1

    for p in percentiles:
      tts_stats_per_i[-1].update({f"{p}": np.percentile(tts_i*tts_factor[i], p)})

  # Statistics of TTS across all instances
  tts_list_percentiles = np.empty((len(percentiles), bs_size))
  for b in range(bs_size):
    new_I = rs.choice(range(len(energy_arrays)), len(energy_arrays), replace=True)

    pos_samples = [beta.rvs(successes[i] + 0.5, failures[i] + 0.5, random_state = rs) for i in new_I]
    r_samples = [np.log(0.01)/np.log(1.0 - np.clip(pos, 1e-15, 1.0)) for pos in pos_samples]
    tts_samples = [tts_factor[new_I[i]] if r < 1 else tts_factor[new_I[i]]*r for i, r in enumerate(r_samples)]

    for pi, p in enumerate(percentiles):
      tts_list_percentiles[pi, b] = np.percentile(tts_samples, p)

  return tts_stats_per_i, \
         suc_stats_per_i, \
         np.mean(tts_list_percentiles, axis=1), \
         np.std(tts_list_percentiles, axis=1)


def sa_instance_task(problem, 
                     track_stats_freq,
                     seed):

  N = problem["n"]

  trace_size = problem["nsweeps"]//track_stats_freq

  rs  = np.random.RandomState(seed)
  x_start = rs.binomial(1, 0.5, N).astype(np.int32)
  
  final_x, min_x, ar_trace = CI.run_mcmc_sat(
    final_x_size = N, 
    min_x_size = N,
    ar_trace_size = trace_size,
    x_start = x_start,
    mode = problem["mode"],
    J_indices = problem["Jidx_sep"], 
    J_values = problem["Jval"],    
    h = problem["h"],
    seed = seed,
    total_sweeps = problem["nsweeps"],
    initial_T = problem["Ti"],
    final_T = problem["Tf"],
    T_schedule = problem["schedules"]["t_schedule"],

    aux_size = problem["aux_size"] if problem["mode"] != "pubo" else 0,
    track_stats_freq = track_stats_freq
  )

  finalE = CI.run_energy_sat(final_x, 
                             J_indices = problem["Jidx_sep"], 
                             J_values = problem["Jval"],    
                             h = problem["h"])
  
  final_energy_gap = finalE - problem["known_gm"]

  minE = CI.run_energy_sat(min_x, 
                          J_indices = problem["Jidx_sep"], 
                          J_values = problem["Jval"],    
                          h = problem["h"])

  min_energy_gap = minE - problem["known_gm"]

  if minE > finalE or minE < problem["known_gm"]:
    print(f"{minE}_{finalE}_{problem['known_gm']}")
    raise RuntimeError("Wrong min energy!")

  log = f"seed = {seed}, final E = {final_energy_gap}, min_E = {min_energy_gap}"

  return min_energy_gap, final_energy_gap, ar_trace, log, problem["id"]


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  
  parser.add_argument("-name", "--name", 
                      help="SAT instance name", 
                      default="default")
  
  parser.add_argument("-N", "--N", 
                      help="SAT problem size", 
                      default="default")
  
  parser.add_argument("-Ti", "--Ti", 
                      help="Initial annealing temperature", 
                      default="default")
  
  parser.add_argument("-Tf", "--Tf", 
                      help="Final annealing temperature", 
                      default="default")
  
  parser.add_argument("-seed", "--seed", 
                      help="random seed", 
                      default="default")
  
  parser.add_argument("-mapping_name", "--mapping_name", 
                      help="mapping_name", 
                      default="default")

  parser.add_argument("-mapping_type", "--mapping_type", 
                      help="mapping_type", 
                      default="default")
  
  parser.add_argument("-mode", "--mode", 
                      help="Solver mode: pubo, qubo", 
                      default="default")

  parser.add_argument("-nsweeps", "--nsweeps", 
                      help="SA sweep number", 
                      default="default")
  
  parser.add_argument("-nreps", "--nreps", 
                      help="SA number of repetitions", 
                      default="default")
  
  parser.add_argument("-inst_a", "--inst_a", 
                      help="Instance [a, ...", 
                      default="default")
  
  parser.add_argument("-inst_b", "--inst_b", 
                      help="Instance ..., b]", 
                      default="default")
  
  parser.add_argument("-num_workers", "--num_workers",
                      help="Number of cpu workers",
                      default="default")
  
  args = parser.parse_args()
  with open(os.path.join('configs', 'default_config.json'), 'r') as f:
    default_config = json.load(f)

  seed = default_config["seed"] if args.seed == "default" else int(args.seed)
  name = default_config["name"] if args.name == "default" else args.name

  mode = default_config["solver_config"]["mode"] if args.mode == "default" else args.mode
  
  assert mode in ["pubo", "qubo"], "given mode is not supported"
  if mode == "pubo":
    mapping_name = ""
    mapping_type = ""
  
  if mode != "pubo":
    mapping_name = default_config["solver_config"]["mapping_name"] if args.mapping_name == "default" else args.mapping_name
    mapping_type = default_config["solver_config"]["mapping_type"] if args.mapping_type == "default" else args.mapping_type
  
  nsweeps = default_config["solver_config"]["nsweeps"] if args.nsweeps == "default" else int(args.nsweeps)
  nreps = default_config["nreps"] if args.nreps == "default" else int(args.nreps)

  percentiles = default_config["percentiles"]
  
  with open(os.path.join('configs', name, 'config.json'), 'r') as f:
    problem_config = json.load(f)
  
  logging.basicConfig(level=logging.WARN, force=True)

  default_config["solver_config"]["energy_gap"] = problem_config["energy_gap"]

  instances_read = problem_config["instances"]
  
  N = problem_config["N"] if args.N == "default" else int(args.N)
  Ti  = problem_config["Ti"] if args.Ti == "default" else float(args.Ti)
  Tf  = problem_config["Tf"] if args.Tf == "default" else float(args.Tf)

  if problem_config["instances_source"] == "range":
    inst_a = instances_read[0] if args.inst_a == "default" else int(args.inst_a)
    inst_b = instances_read[1] if args.inst_b == "default" else int(args.inst_b)

    instances_list = list(range(inst_a, inst_b+1))
    
  elif problem_config["instances_source"] == "list":
    instances_list = instances_read

  problems = []
  result = {}

  logging.info(f"name: {name}, native size = {N}")
  for idx, i in enumerate(instances_list):
    if mode == "pubo":
      read_instance = cnf_reader_to_p(os.path.join('..', 'problems', name, f'{name}{N}-0{i}.cnf'), 
                                      0, 
                                      "pubo")
      J = read_instance[0][0]
      h = read_instance[0][1].astype(np.int32)

      known_global_min = int(read_instance[1])
      # cnf_clauses = read_instance[2]

    else:
      W, h, C, aux_size = qubo_sat_map({"mapping_name": mapping_name, 
                                        "mapping_type": mapping_type,
                                        "instance": os.path.join('..', 'problems', name, f'{name}{N}-0{i}.cnf')})
      
      
      nnz_indices = []
      nnz_values = []
      for row, row_v in enumerate(W):
        for col, col_v in enumerate(row_v[row + 1:]):
          if col_v != 0:
            nnz_indices.append([row, row + 1 + col])
            nnz_values.append(-col_v)

      J = [{"indices": np.stack(nnz_indices).transpose().astype(np.int32), 
            "values": np.array(nnz_values).astype(np.int32)}]
      h = -h.astype(np.int32)
      known_global_min = -int(C.item())

    logging.info(f"Instance {i} read. N = {h.size}")
    

    Jidx_sep = np.empty(0, dtype=np.int32)
    for sj in J:
      Jidx_sep = np.append(Jidx_sep, 
                          np.concatenate([np.transpose(sj["indices"]).astype(np.int32), 
                                          np.full((sj["indices"].shape[1], 1), -1)], axis=1, dtype=np.int32).flatten())
    Jval = np.concatenate([sj["values"] for sj in J]).astype(np.int32)

    problems.append({
      "mode": mode,
      "id": i,

      "n": h.size,

      "known_gm": known_global_min,

      "nsweeps": nsweeps,

      "Ti": Ti,
      "Tf": Tf,

      "schedules": problem_config["schedules"],

      "Jidx_sep": Jidx_sep,
      "Jval": Jval,
      "h": h
    })

    if mode != "pubo":
      problems[-1].update({"aux_size": aux_size})
      
    result.update(
      {i: {
        "known_gm": known_global_min,

        "energy_gap": [],

        "ar_trace": [],
        "ar_trace_native": [],
        "energy_trace": [],
        "misflip_trace": [],

        "sign_mismatch": [],
        "de_mismatch": [],
        "rank_correlation": []}
      }
    )

  logging.info(f"Solver mode: {default_config['solver_config']['mode']}")
  start_time = time.time()

  num_workers = default_config["num_workers"] if args.num_workers == "default" else int(args.num_workers)

  if num_workers > 1: #run parallel process for each instance

    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers = num_workers) as executor:
      with tqdm(total = nreps*len(problems)) as progress:
        
        futures = []

        for n in range(nreps):
          for j, problem in enumerate(problems):
            future = executor.submit(sa_instance_task, 
                                     problem,
                                     default_config["track_stats_freq"],
                                     seed + n + j)
            
            future.add_done_callback(lambda p: progress.update())
            futures.append(future)

        ar_all = []
        ar_all_pubo = []
        for future in concurrent.futures.as_completed(futures): 
          energy_gap, final_energy_gap, ar_trace, log, id = future.result()

          ar_all.append(ar_trace.mean())
          logging.info(f" {name}{id} | AR={ar_trace.mean():1.4f} | " + log)

          result[id]["energy_gap"].append(energy_gap)

  else:
    ar_all = []
    for n in range(nreps):
      for j, problem in enumerate(problems):
        energy_gap, final_energy_gap, ar_trace, log, id = sa_instance_task(problem, 
                                                                          default_config["track_stats_freq"],
                                                                          seed + n + j)
          
        ar_all.append(ar_trace.mean())
        logging.info(f" {name}{id} | AR={ar_trace.mean():1.4f} | "+ log)

        result[id]["energy_gap"].append(energy_gap)
  

  total_time = time.time() - start_time

  problem_name = f"{name}_{N}"
  folder_name = f"{mode}_{seed}"

  os.makedirs("results", exist_ok=True)
  os.makedirs(os.path.join('results', problem_name), exist_ok=True)
  os.makedirs(os.path.join('results', problem_name, folder_name), exist_ok=True)
  
  energies = [result[p["id"]]["energy_gap"] for p in problems]
  sweep_sizes = np.array([p["n"] for p in problems])

  tts_per_instance,\
  suc_per_instance,\
  tts_percentiles,\
  tts_percentiles_sigma = tts_bootstrap(energies, 
                                        tts_factor = sweep_sizes,
                                        success_energy_gap = default_config["solver_config"]["energy_gap"], 
                                        seed = seed,
                                        bs_size = 10000,
                                        percentiles=percentiles)
  

  print_tts = []
  key = "50"
  for i, tts_i in enumerate(tts_per_instance):
    print_tts.append(nsweeps*tts_i[key])

  print(f"avg ar_trace: {np.mean(ar_all)} +- {np.std(ar_all)}")

  print(f"succ (out of {nreps}): {suc_per_instance}")
  print(f"{name}{N}_{Ti}_{Tf}_{mode}_{mapping_name}_{mapping_type}_{nsweeps}_{nreps} | TTS_0.5i_0.5 = {np.median(print_tts)}")

  for p, pm, ps in zip(percentiles, tts_percentiles, tts_percentiles_sigma):
    if int(p) == 50:
      print(f"<TTS_i>{p} mean +- sigma = {nsweeps*pm} +- {nsweeps*ps}")

  if (default_config["log_to_file"]):
    with open(os.path.join('results', problem_name, folder_name, f"{name}{N}_{Ti}_{Tf}_{mode}_{mapping_name}_{mapping_type}_{nsweeps}_{nreps}_{instances_list[0]}_{instances_list[-1]}.dat"), 'w') as f:

      print(f"Instances: {instances_list} | repertitions: {nreps}, total time: {total_time}", file=f)
      print(f"succ (out of {nreps}): {suc_per_instance}", file=f)
      print(f"avg ar_trace: {np.mean(ar_all)} +- {np.std(ar_all)}", file=f)

      print(f"TTS_0.5i_0.5 = {np.median(print_tts)}", file=f)

      for p, pm, ps in zip(percentiles, tts_percentiles, tts_percentiles_sigma):
        np.savetxt(f, [nsweeps*pm, nsweeps*ps], fmt= '%.5f',  delimiter=' ', header=f"<TTS_i>{p} mean +- sigma")
        
      for i, p in enumerate(problems):
        save_tts = []
        for key, value in tts_per_instance[i].items():
          save_tts.append([int(key)/100, value*nsweeps])

        np.savetxt(f, np.array(save_tts, dtype=float), 
                  fmt= '%.5f',  
                  delimiter=' ', 
                  header=f"{name}{p['id']}")
          
