import numpy as np
from typing import Tuple, Union, Any, List, Dict, Literal

import torch as th

import os

class genericReader:
   def __init__(self, 
                directory: str, 
                problemName: str, 
                pmode: str):
      self.directory = directory
      self.problemName = problemName
      self.pmode = pmode #pubo or pising

   def get(self, 
           instNames: List[int], 
           N: int):

      read_instances = []
      for i in instNames:
         read_i =  cnf_reader_to_p(os.path.join(self.directory, "data", f"{self.problemName}_{i}.cnf"), 
                                    N, 
                                    self.pmode)
         read_instances.append(read_i)
         
      return read_instances


def unroll_clause_to_polynomial(monomial: Tuple[float, List[Tuple[str, int]]], 
                                polynomial: Dict, constant: List[int],
                                pmode):
   if (monomial[1] == [('s', abs(m[1])) for m in monomial[1]] and pmode == 'pising'): 

      polynomial[len(monomial[1])]["indices"].append([m[1]-1 for m in monomial[1]])
      polynomial[len(monomial[1])]["values"].append(monomial[0])

   elif (monomial[1] == [('b', abs(m[1])) for m in monomial[1]] and pmode == 'pubo'): 

      polynomial[len(monomial[1])]["indices"].append([m[1]-1 for m in monomial[1]])
      polynomial[len(monomial[1])]["values"].append(monomial[0])
   
   elif(monomial[1][0][0] == 'c'):
      constant[0] += 1
   
   else:
      for i, m in enumerate(monomial[1]):
         if m[0] == 'x':
            if pmode == 'pising':
               new_monomial_a = (monomial[0]*np.sign(m[1]), monomial[1].copy())
               new_monomial_a[1][i] = ('s', abs(m[1]))
               
               new_monomial_b = (monomial[0], [])
               new_monomial_b[1].extend(monomial[1][:i])

               if i != len(monomial[1])-1:
                  new_monomial_b[1].extend(monomial[1][i+1:])
               
               if len(new_monomial_b[1]) == 0:
                  new_monomial_b[1].append(('c', 0))
               
               unroll_clause_to_polynomial(new_monomial_a, polynomial, constant, pmode)
               unroll_clause_to_polynomial(new_monomial_b, polynomial, constant, pmode)

            elif pmode == 'pubo':
               new_monomial_a = (monomial[0]*np.sign(m[1]), monomial[1].copy())
               new_monomial_a[1][i] = ('b', abs(m[1]))

               unroll_clause_to_polynomial(new_monomial_a, polynomial, constant, pmode)

               if np.sign(m[1]) < 0:
                  new_monomial_b = (monomial[0], [])
                  new_monomial_b[1].extend(monomial[1][:i])

                  if i != len(monomial[1])-1:
                     new_monomial_b[1].extend(monomial[1][i+1:])
                  
                  if len(new_monomial_b[1]) == 0:
                     new_monomial_b[1].append(('c', 0))
               
                  unroll_clause_to_polynomial(new_monomial_b, polynomial, constant, pmode)

            else:
               raise RuntimeError("Wrong pmode!")

            break

# (1-xa)(1-xb)(xc)(1-xd) = (1-sa)(1-sb)(1+s)(1-sd)/16
# x = (1+s)/2

def cnf_reader_to_p(txtfile: str, 
                    N: int, 
                    pmode: str) -> Tuple[Tuple[List[th.sparse_coo_tensor], np.ndarray], float, np.ndarray]:
   sJ = []

   ground_state = 0.0
   polynomial = {}

   all_clauses = np.empty(0, dtype=np.int32)

   with open(txtfile, 'r') as f:
      for line in f:
         if line.startswith('c') or line.startswith('p') or line.startswith('\n'):
            continue

         L = np.fromstring(line, dtype=int, sep=' ')
         all_clauses = np.append(all_clauses, L)

         for k in range(1, len(L)):
            if k not in polynomial:
               polynomial.update({k: {"indices": [], "values": []}})

         if np.max(np.abs(L)) > N:
            N = int(np.max(np.abs(L)))
         
         constant = [0]
         if pmode =='pising':
            unroll_clause_to_polynomial((pow(2, -len(L)+1), [('x', -l) for l in L[:-1]]), 
                                       polynomial, constant, pmode)

            ground_state += constant[0]*pow(2, -len(L)+1)

         elif pmode == 'pubo':
            unroll_clause_to_polynomial((1.0, [('x', -l) for l in L[:-1]]), 
                                          polynomial, constant, pmode)

            ground_state += constant[0]
         else: 
            raise RuntimeError("Wrong pmode")

   h = np.zeros(N, dtype=float)
   for key, value in sorted(polynomial.items(), reverse=True):
      if key != 1:
         sp_tensor = th.sparse_coo_tensor(np.transpose(np.sort(np.array(value["indices"]), axis=1)), 
                                          value["values"], 
                                          size = tuple(np.full(key, N))).coalesce()
         if th.min(sp_tensor.values().abs()) == 0:
            nnz_indices = []
            nnz_values = []

            for i, v in zip(sp_tensor.indices().transpose(0, 1), sp_tensor.values()):
               if v != 0:
                  nnz_indices.append(i)
                  nnz_values.append(v)
            
            sJ.append({"indices": np.stack(nnz_indices).transpose(), "values": -np.array(nnz_values)})
         else:
            sJ.append({"indices": sp_tensor.indices().numpy(), "values": -sp_tensor.values().numpy()})

      else: 
         for i,v in zip(value["indices"], value["values"]):
            h[i[0]] -= v

   
   return (sJ, h), -ground_state, all_clauses.flatten().astype(np.int32)