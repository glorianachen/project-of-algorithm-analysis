import sys, getopt
from os.path import isfile
from math import sqrt
import argparse
import numpy as np
import time
import os
#import sys
#import branchandbound

# elif algorithm == 'Approx':
#      approx_1 = MSTApprox.MSTApprox(self.graph,instance,seed,cutoff)
#      approx_1.generate_tour()

def algorithm(argv):
    inst = sys.argv[2]
    alg = sys.argv[4]
    time_limit = int(sys.argv[6])
    print("file name: ", inst)
    sub = inst.rfind('/')
    if sub == -1:
        city_name = inst[:-4]
    else:
        city_name = inst[(sub+1):-4]
    print("sub: ", sub)
    print("city name: ", city_name)

    # python main.py -inst DATA/small.tsp -alg BnB -time 10
    # translate city instance to 2D adjacent matrix G
    if not isfile(inst):
        print("File not found. Check the city name!")
        sys.exit()
    f=open(inst)
    line=f.readline()
    while not 'NODE_COORD_SECTION\n' in line:
      line=f.readline()
    line=f.readline()
    city={}
    while line!='EOF\n':
      v = line.split(' ')
      city[int(v[0])] = [float(v[1]), float(v[2])]
      line=f.readline()
    #print("city: ",city)

    # calculate euclidean distance, store in 2D adjacent matrix
    rows, cols = (len(city), len(city)) 
    G = [[0 for x in range(cols)] for y in range(rows)]  
    for key_v, value_v in city.items():
        for key_u, value_u in city.items():
            G[key_v - 1][key_u - 1] = round(np.linalg.norm(np.array(value_v) - np.array(value_u)))
    #print("G: ",G)

    start = time.time()
    cost = np.inf
    trace = []
    solution = []

    # Step1: get the MST using Prim's
    T_Path = []
    T = np.full_like(G, np.inf)
    T, T_Path = prim(G, 0)
    print("MST:", T)
    print("T_Path: ",T_Path)
    print()

    #Go through each case of selecting a vertex as the root node
    for i in range(len(city)):
        # # Step1: get the MST using Prim's
        # T_Path = []
        # T = np.full_like(G, np.inf)
        # T, T_Path = prim(G, i)
        # print("MST:", T)
        # print("T_Path: ",T_Path)

        # Step2: traverse the MST in preorder walk
        T_walk = []
        T_walk = preorder_tree_walk(T, i)
        print("T_walk: ",T_walk)
        
        # Step3: get Hamiltonian cycle
        H = np.full_like(G, np.inf)
        H_Path = []
        H, H_Path, cur_cost = create_H(G, T_walk, i)
        #print("H:", H)
        #print("H_Path:", H_Path)
        #print("cur_cost:", cur_cost)
        
        if cur_cost < cost:
            cost = cur_cost
            trace_time = (time.time() - start)
            solution = T_walk 
            print("Update")
            print("T_walk: ",T_walk)    
            print("trace_time: ", trace_time)
            print("cost:", cost)
            # update trace file information
            trace.append((trace_time, cost))
            
        if (time.time() - start) > time_limit:
            print("break!!!!")
            break
    print()
    print("Final:")
    print(cost)
    print(solution)
    print(trace)

    #write into file
    if 'output' not in os.listdir('./'):
        os.mkdir('./output')
    with open('output/'+city_name + "_Approx_" + str(time_limit) +'.trace', 'w') as f:
        for (a, b) in trace:
            f.write('{:.2f}, {}\n'.format(a, b))
            # f.write(str(a) + ', ' + str(b))
            # f.write('\n') 
    with open('output/'+city_name + "_Approx_" + str(time_limit) +'.sol', 'w') as f:
        f.write('{}\n'.format(cost))
        for vertex in solution[:-1]:
            f.write(str(vertex) + ',') 
        f.write(str(solution[-1])) 


# compute MST using Prim
def prim(G, root_index):
    visited_ids = [root_index] # initialize the set of visited nodes
    T_Path = []
    while len(visited_ids) != len(G[0]):
        no_visited_ids = contains_no_visited_ids(G, visited_ids) # maintain the set of non-visited nodes
        (min_from, min_to), min_weight = find_min_edge(G, visited_ids, no_visited_ids)
        visited_ids.append(min_to) # maintain the set of visited nodes
        T_Path.append((min_from, min_to))
    T = np.full_like(G, np.inf) # the matrix form of MST, consisting (n-1) edges
    for (from_, to_) in T_Path:
        T[from_][to_] = G[from_][to_]
        T[to_][from_] = G[to_][from_]
    return T, T_Path

# maintain non-visited nodes set
def contains_no_visited_ids(G, visited_ids):
    no_visited_ids = []
    [no_visited_ids.append(idx) for idx, _ in enumerate(G) if idx not in visited_ids]
    return no_visited_ids

# Find cheapest edge, add to visited nodes (Greedy)
def find_min_edge(G,visited_ids, no_visited_ids):
    min_weight, min_from, min_to = np.inf, np.inf, np.inf
    for from_index in visited_ids:
        for to_index, weight in enumerate(G[from_index]):
            if from_index != to_index and weight < min_weight and to_index in no_visited_ids:
                min_to = to_index
                min_from = from_index
                min_weight = G[min_from][min_to]
    return (min_from, min_to), min_weight

# Preorder the MST
def preorder_tree_walk(T, root_index):
    is_visited = [False] * T.shape[0]
    stack = [root_index]
    T_walk = []
    while len(stack) != 0:
        node = stack.pop()
        T_walk.append(node)
        is_visited[node] = True
        nodes = np.where(T[node] != np.inf)[0]
        if len(nodes) > 0:
            [stack.append(node) for node in reversed(nodes) if is_visited[node] is False]
    return T_walk

# generate Hamliton Cycle H
def create_H(G, L, root_index):
    cost = 0
    H = np.full_like(G, np.inf)
    H_Path = []
    for i, from_node in enumerate(L[0:-1]):
        to_node = L[i + 1]
        H[from_node][to_node] = G[from_node][to_node]
        H[to_node][from_node] = G[to_node][from_node]
        H_Path.append((from_node, to_node))
        cost = cost + G[from_node][to_node]
    H_Path.append((to_node, root_index)) # add one last edge to form a cycle
    cost = cost + G[to_node][root_index]
    return H, H_Path, int(cost)

if __name__ == "__main__":
   algorithm(sys.argv)