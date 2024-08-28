import numpy as np
import os
import typing as t

def load_clauses_from_cnf(file_path: str) -> t.List[t.List[int]]:
    with open(file_path, 'r') as file:
        lines = file.readlines()
        clauses = []
        for line in lines:
            if line.startswith('c') or line.startswith('p') or line.startswith('\n'):
                continue
            elif line.startswith('%'):
                break
            clause = [int(x) for x in line.strip().split() if x != '0']
            clauses.append(clause)
    # clean empty clauses
    clauses = [l for l in clauses if l]
    return clauses

# QUBO mappings available

# Returns W, B, C of the QUBO energy from 3/4-SAT cnf (k = 3/4 SAT only so far)
# W is returned as a symmetric matrix, i.e. E = x^T*W*x/2 + Bx + C
    
# mapping type 1: "clause_wise" mapping introduces an auxiliary variable for each clause of the cnf (Total QUBO size N = N_var + N_clauses)
# mapping type 2: "shared" mapping introduces each auxiliary varibale for multiple clauses by solving an optimization 
#     problem of finding x_ix_j = y substitutions minimizing the total number of aux variables (Total QUBO size N < N_var + N_clauses)
#
# mapping name 1: "rosenberg" penalty (observed to be better for parallel updates on uf 3-SAT)
# mapping name 2: "kzfd" penalty      (observed to be better for single updates on uf 3-SAT)

def qubo_sat_map(config: t.Dict) -> t.Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    
    instance_name = config["instance"]
    clauses = load_clauses_from_cnf(instance_name)

    num_vars = np.max(abs(np.concatenate(clauses, axis=0)))

    clauses_1 = []
    clauses_2 = []
    clauses_34 = []
    use_4sat = False

    for c in clauses:
        if len(c) > 4:
            raise RuntimeError("max 4 sat only!")

        elif len(c) == 4:
            use_4sat = True
            clauses_34.append(c)

        elif len(c) == 3:
            clauses_34.append(c)

        elif len(c) == 2:
            clauses_2.append(c)

        elif len(c) == 1:
            clauses_1.append(c)

    mapping_type = config["mapping_type"]   #clause_wise or shared   
    if not (mapping_type in ["clause_wise", "shared"]):
        raise RuntimeError("wrong mapping type!")
    
    mapping_name = config["mapping_name"]   #rosenberg or kzfd    
    if not (mapping_name in ["rosenberg", "kzfd"]):
        raise RuntimeError("wrong mapping name!") 
    
    if use_4sat:
        W, B, C = qubo_4sat_map(num_vars, clauses_34, mapping_name, mapping_type)
    else:
        if mapping_type == "clause_wise":
            W, B, C = clause_wise_qubo_3sat_map(num_vars, clauses_34, mapping_name)
        else:
            W, B, C = shared_qubo_3sat_map(num_vars, clauses_34, mapping_name)
    
    for c in clauses_1:
        if c[0] < 0:
            B[-c[0]-1] += 1
        else:
            B[c[0]-1] += -1
            C += 1

    for c in clauses_2:
        if c[0] < 0 and c[1] < 0:
            W[-c[0]-1, -c[1]-1] += 1
            W[-c[1]-1, -c[0]-1] += 1

        elif c[0] > 0 and c[1] < 0:
            W[c[0]-1, -c[1]-1] += -1
            W[-c[1]-1, c[0]-1] += -1
            B[-c[1]-1] += 1

        elif c[0] < 0 and c[1] > 0:
            W[-c[0]-1, c[1]-1] += -1
            W[c[1]-1, -c[0]-1] += -1
            B[-c[0]-1] += 1

        else:
            W[c[0]-1, c[1]-1] += 1
            W[c[1]-1, c[0]-1] += 1

            B[c[0]-1] += -1
            B[c[1]-1] += -1

            C += 1
    
    return W, B, C, (B.shape[0] - num_vars).item()


def shared_qubo_3sat_map(num_vars: int, 
                         clauses: t.List[t.List[int]], mapping_name) -> t.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    num_clauses = len(clauses)

    cl = -np.array(clauses)            #invert the clauses to map to energy
    cl_idx =  np.abs(cl) - np.full_like(cl, 1) #count the variables from 0

    W1 = np.zeros((num_vars, num_vars))
    B1 = np.zeros(num_vars)
    C = np.zeros(1)
    
    for i in range(num_clauses):
        if cl[i, 0] > 0: 
            if cl[i, 1] > 0: 
                if cl[i, 2] > 0:
                    pass
                else:
                    W1[cl_idx[i,0], cl_idx[i,1]] += 1

            else:
                if cl[i, 2] > 0:
                    W1[cl_idx[i,0], cl_idx[i,2]] += 1
                else:
                    W1[cl_idx[i,0], cl_idx[i,1]] -= 1
                    W1[cl_idx[i,0], cl_idx[i,2]] -= 1

                    B1[cl_idx[i,0]] += 1
            
        else:
            if cl[i,  1] > 0: 
                if cl[i, 2] > 0: 
                    W1[cl_idx[i,1], cl_idx[i,2]] += 1
                else:
                    W1[cl_idx[i,0], cl_idx[i,1]] -= 1
                    W1[cl_idx[i,1], cl_idx[i,2]] -= 1

                    B1[cl_idx[i,1]] += 1
                
            else:
                if cl[i,2] > 0: 
                    W1[cl_idx[i,0], cl_idx[i,2]] -= 1
                    W1[cl_idx[i,1], cl_idx[i,2]] -= 1

                    B1[cl_idx[i,2]] += 1
                else:
                    W1[cl_idx[i,0], cl_idx[i,1]] += 1
                    W1[cl_idx[i,0], cl_idx[i,2]] += 1
                    W1[cl_idx[i,1], cl_idx[i,2]] += 1

                    B1[cl_idx[i,0]] -= 1
                    B1[cl_idx[i,1]] -= 1
                    B1[cl_idx[i,2]] -= 1

                    C += 1 
    
    count3 = np.full(num_clauses, 1)

    #take into account repeating 3rd order terms
    for i in range(num_clauses):
        if count3[i] == 0:
            continue
        c1 = np.sort(cl_idx[i, :])

        first_match = True
        for j in range(i+1, num_clauses):
            if count3[j] == 0:
                continue
            c2 = np.sort(cl_idx[j, :])
            if (c1 == c2).all():
                if first_match:
                    if np.prod(cl[i, :]) > 0:
                        count3[i] = 1
                    else:
                        count3[i] = -1
                    first_match = False
                
                if np.prod(cl[j, :]) > 0:
                    count3[i] += 1
                else:
                    count3[i] -= 1
                
                count3[j] = 0

    pairs_matrix = np.zeros((num_vars, num_vars), dtype=int)
    for i in range(num_clauses):
        if count3[i] != 0:
            pairs_matrix[cl_idx[i, 0], cl_idx[i, 1]] += 1
            pairs_matrix[cl_idx[i, 0], cl_idx[i, 2]] += 1
            pairs_matrix[cl_idx[i, 1], cl_idx[i, 2]] += 1
    
    pairs_matrix += np.transpose(pairs_matrix)

    pair_count = []
    for i in range(num_vars):
        for j in range(i+1, num_vars):
            if pairs_matrix[i, j] > 0:
                pair_count.append([np.array([i, j]), pairs_matrix[i, j]])

    pair_count = sorted(pair_count, key = lambda p: p[1], reverse=True)

    shared_pairs = []
    track_clauses = np.full(num_clauses, 1)

    for i in range(num_clauses):
        if i == 45:
            pass
        if track_clauses[i] == 0 or count3[i] == 0:
            continue
        
        c01 = np.sort([cl_idx[i, 0], cl_idx[i, 1]])
        c02 = np.sort([cl_idx[i, 0], cl_idx[i, 2]])
        c12 = np.sort([cl_idx[i, 1], cl_idx[i, 2]])

        for k in range(len(pair_count)):
            save_pair = False

            if (c01 == pair_count[k][0]).all():
                save_pair = True
                for l in range(len(pair_count)):
                    if((c02 == pair_count[l][0]).all() or (c12 == pair_count[l][0]).all()):
                        pair_count[l][1] -= 1

            if (c02 == pair_count[k][0]).all():
                save_pair = True
                for l in range(len(pair_count)):
                    if((c01 == pair_count[l][0]).all() or (c12 == pair_count[l][0]).all()):
                        pair_count[l][1] -= 1
                        
            if (c12 == pair_count[k][0]).all():
                save_pair = True
                for l in range(len(pair_count)):
                    if((c01 == pair_count[l][0]).all() or (c02 == pair_count[l][0]).all()):
                        pair_count[l][1] -= 1

            if save_pair:
                pair_count[k][1] -= 1
                
                for j in range(i+1, num_clauses):
                    if count3[j] == 0 or track_clauses[j] == 0:
                        continue

                    c01 = np.sort([cl_idx[j, 0], cl_idx[j, 1]])
                    c02 = np.sort([cl_idx[j, 0], cl_idx[j, 2]])
                    c12 = np.sort([cl_idx[j, 1], cl_idx[j, 2]])

                    if (c01 == pair_count[k][0]).all():
                        for l in range(len(pair_count)):
                            if((c02 == pair_count[l][0]).all() or (c12 == pair_count[l][0]).all()):
                                pair_count[l][1] -= 1
                        pair_count[k][1] -= 1
                        track_clauses[j] = 0

                    if (c02 == pair_count[k][0]).all():
                        for l in range(len(pair_count)):
                            if((c01 == pair_count[l][0]).all() or (c12 == pair_count[l][0]).all()):
                                pair_count[l][1] -= 1
                        pair_count[k][1] -= 1
                        track_clauses[j] = 0

                    if (c12 == pair_count[k][0]).all():
                        for l in range(len(pair_count)):
                            if((c01 == pair_count[l][0]).all() or (c02 == pair_count[l][0]).all()):
                                pair_count[l][1] -= 1
                        pair_count[k][1] -= 1
                        track_clauses[j] = 0

                shared_pairs.append(pair_count[k][0])
                if pair_count[k][1] != 0:
                    raise RuntimeError("Error")
                
                pair_count = sorted(pair_count, key = lambda p: p[1], reverse=True)

                break
    
    N = num_vars + len(shared_pairs)

    W = np.zeros((N, N))
    B = np.zeros(N)

    W[:num_vars, :num_vars] = W1
    B[:num_vars] = B1

    if mapping_name == "rosenberg":
        penalty_lb = np.zeros((len(shared_pairs), 3))

    check = 0
    for i in range(num_clauses):
        c01 = np.sort([cl_idx[i, 0], cl_idx[i, 1]])
        c02 = np.sort([cl_idx[i, 0], cl_idx[i, 2]])
        c12 = np.sort([cl_idx[i, 1], cl_idx[i, 2]])
    
        found_pair = False
        for k in range(len(shared_pairs)):
            if (c01 == shared_pairs[k]).all():
                found_pair = True
                if np.prod(cl[i, :]) > 0:
                    W[cl_idx[i, 2], num_vars + k] += 1
                else:
                    W[cl_idx[i, 2], num_vars + k] -= 1

            if (c02 == shared_pairs[k]).all():
                found_pair = True
                if np.prod(cl[i, :]) > 0:
                    W[cl_idx[i, 1], num_vars + k] += 1
                else:
                    W[cl_idx[i, 1], num_vars + k] -= 1

            if (c12 == shared_pairs[k]).all():
                found_pair = True
                if np.prod(cl[i, :]) > 0:
                    W[cl_idx[i, 0], num_vars + k] += 1
                else:
                    W[cl_idx[i, 0], num_vars + k] -= 1
    
            if found_pair:
                check += 1
                if mapping_name == "rosenberg":
                    if np.prod(cl[i, :]) > 0:
                        penalty_lb[k, 0] += 1
                    else:
                        penalty_lb[k, 1] += 1
                    
                    if penalty_lb[k, 2] <  np.max(penalty_lb[k, :]):
                        W[shared_pairs[k][0], shared_pairs[k][1]] += 1
                        W[shared_pairs[k][0], num_vars + k] -= 2
                        W[shared_pairs[k][1], num_vars + k] -= 2
                        B[num_vars + k] += 3

                        penalty_lb[k, 2] += 1

                elif mapping_name == "kzfd":
                    W[shared_pairs[k][0], shared_pairs[k][1]] += 1
                    W[shared_pairs[k][0], num_vars + k] -= 1
                    W[shared_pairs[k][1], num_vars + k] -= 1
                    B[num_vars + k] += 1
                    
                    if np.prod(cl[i, :]) < 0:
                        B[num_vars + k] += 1
                        W[shared_pairs[k][0], shared_pairs[k][1]] -= 1

                else:
                    raise RuntimeError("Unknown mapping!")
                
                break
    # print(f"substituted {check} clauses")
    W += np.transpose(W)

    return W, B, C


def clause_wise_qubo_3sat_map(num_vars: int, 
                              clauses: t.List[t.List[int]], mapping_name) -> t.Tuple[np.ndarray, np.ndarray, np.ndarray]:

    num_clauses = len(clauses)

    N = num_vars + num_clauses

    cl = -np.array(clauses)            #invert the clauses to map to energy
    cl_idx =  np.abs(cl) - np.full_like(cl, 1) #count the variables from 0

    W = np.zeros((N, N))
    B = np.zeros(N)
    C = np.zeros(1)

    for i in range(num_clauses):
      if cl[i, 2] > 0:
        W[num_vars + i, cl_idx[i, 2]] = 1
      else:
        B[num_vars + i] = 1
        W[num_vars + i, cl_idx[i, 2]] = -1

    for i in range(num_clauses):
      if mapping_name == "rosenberg":
          B[num_vars + i] += 3
      else:
          B[num_vars + i] += 1

      if cl[i, 0] > 0 and cl[i, 1] > 0:
          W[cl_idx[i, 0], cl_idx[i, 1]] += 1

          if mapping_name == "rosenberg":
              W[num_vars + i, cl_idx[i, 0]] -= 2
              W[num_vars + i, cl_idx[i, 1]] -= 2
          else:
              W[num_vars + i, cl_idx[i, 0]] -= 1
              W[num_vars + i, cl_idx[i, 1]] -= 1

      elif cl[i, 0] > 0 and cl[i, 1] < 0:
          W[cl_idx[i, 0], cl_idx[i, 1]] -= 1
          B[cl_idx[i, 0]] += 1

          if mapping_name == "rosenberg":
              W[num_vars + i, cl_idx[i, 0]] -= 2
              W[num_vars + i, cl_idx[i, 1]] += 2
              B[num_vars + i] -= 2
          else:
              W[num_vars + i, cl_idx[i, 0]] -= 1
              W[num_vars + i, cl_idx[i, 1]] += 1
              B[num_vars + i] -= 1

      elif cl[i, 0] < 0 and cl[i, 1] > 0:
          W[cl_idx[i, 0], cl_idx[i, 1]] -= 1
          B[cl_idx[i, 1]] += 1

          if mapping_name == "rosenberg":
              W[num_vars + i, cl_idx[i, 1]] -= 2
              W[num_vars + i, cl_idx[i, 0]] += 2
              B[num_vars + i] -= 2
          else:
              W[num_vars + i, cl_idx[i, 1]] -= 1
              W[num_vars + i, cl_idx[i, 0]] += 1
              B[num_vars + i] -= 1

      else:
          W[cl_idx[i, 0], cl_idx[i, 1]] += 1
          B[cl_idx[i, 0]] -= 1
          B[cl_idx[i, 1]] -= 1
          C += 1  # add the constant

          if mapping_name == "rosenberg":
              W[num_vars + i, cl_idx[i, 0]] += 2
              W[num_vars + i, cl_idx[i, 1]] += 2
              B[num_vars + i] -= 4
          else:
              W[num_vars + i, cl_idx[i, 0]] += 1
              W[num_vars + i, cl_idx[i, 1]] += 1
              B[num_vars + i] -= 2

    W += W.transpose()   

    return W, B, C

# Reduces the 4sat problem to 3sat: xa*xb*xc*xd -> xa*xb*y + penalty and 
# performs quadratization according to the schemes above
def qubo_4sat_map(num_vars: int, 
                  clauses: t.List[t.List[int]], 
                  mapping_name, 
                  mapping_type) -> t.Tuple[np.ndarray, np.ndarray, np.ndarray]:

    cl_3sat = []
    cl_4sat = []

    n_aux = num_vars
    for c in clauses:
        if len(c) == 4:
            n_aux += 1
            cl_4sat.append(-np.array(c))
            cl_3sat.append(np.array([c[0], c[1], -n_aux]))
        else:
            cl_3sat.append(-np.array(c))
    
    N = num_vars + len(cl_4sat)

    W0 = np.zeros((N, N))
    B0 = np.zeros(N)
    C0 = np.zeros(1)

    cl_4idx = np.abs(cl_4sat) - np.full_like(cl_4sat, 1) 

    for i in range(len(cl_4sat)):
      if mapping_name == "rosenberg":
          B0[num_vars + i] += 3
      else:
          B0[num_vars + i] += 1

      if cl_4sat[i][2] > 0 and cl_4sat[i][3] > 0:
          W0[cl_4idx[i, 2], cl_4idx[i, 3]] += 1

          if mapping_name == "rosenberg":
              W0[num_vars + i, cl_4idx[i, 2]] -= 2
              W0[num_vars + i, cl_4idx[i, 3]] -= 2
          else:
              W0[num_vars + i, cl_4idx[i, 2]] -= 1
              W0[num_vars + i, cl_4idx[i, 3]] -= 1

      elif cl_4sat[i][2] > 0 and cl_4sat[i][3] < 0:
          W0[cl_4idx[i, 2], cl_4idx[i, 3]] -= 1
          B0[cl_4idx[i, 2]] += 1

          if mapping_name == "rosenberg":
              W0[num_vars + i, cl_4idx[i, 2]] -= 2
              W0[num_vars + i, cl_4idx[i, 3]] += 2
              B0[num_vars + i] -= 2
          else:
              W0[num_vars + i, cl_4idx[i, 2]] -= 1
              W0[num_vars + i, cl_4idx[i, 3]] += 1
              B0[num_vars + i] -= 1

      elif cl_4sat[i][2] < 0 and cl_4sat[i][3] > 0:
          W0[cl_4idx[i, 2], cl_4idx[i, 3]] -= 1
          B0[cl_4idx[i, 3]] += 1

          if mapping_name == "rosenberg":
              W0[num_vars + i, cl_4idx[i, 3]] -= 2
              W0[num_vars + i, cl_4idx[i, 2]] += 2
              B0[num_vars + i] -= 2
          else:
              W0[num_vars + i, cl_4idx[i, 3]] -= 1
              W0[num_vars + i, cl_4idx[i, 2]] += 1
              B0[num_vars + i] -= 1

      else:
          W0[cl_4idx[i, 2], cl_4idx[i, 3]] += 1
          B0[cl_4idx[i, 2]] -= 1
          B0[cl_4idx[i, 3]] -= 1
          C0 += 1

          if mapping_name == "rosenberg":
              W0[num_vars + i, cl_4idx[i, 2]] += 2
              W0[num_vars + i, cl_4idx[i, 3]] += 2
              B0[num_vars + i] -= 4
          else:
              W0[num_vars + i, cl_4idx[i, 2]] += 1
              W0[num_vars + i, cl_4idx[i, 3]] += 1
              B0[num_vars + i] -= 2

    W0 += W0.transpose()
    
    if mapping_type == "clause_wise":
        W, B, C = clause_wise_qubo_3sat_map(N, cl_3sat, mapping_name)
    else:
        W, B, C = shared_qubo_3sat_map(N, cl_3sat, mapping_name)

    W[:N, :N] += W0
    B[:N] += B0
    C += C0

    return W, B, C