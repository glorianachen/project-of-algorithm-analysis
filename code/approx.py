import sys, getopt
from os.path import isfile
from math import sqrt
import argparse
import numpy as np
import time
import os


class Approx:
	def __init__(self, city, timelimit, seed):
		# calculate euclidean distance, store in 2D adjacent matrix
		rows, cols = (len(city), len(city)) 
		G = [[0 for x in range(cols)] for y in range(rows)]  
		for key_v, value_v in city.items():
		    for key_u, value_u in city.items():
		        G[key_v - 1][key_u - 1] = round(np.linalg.norm(np.array(value_v) - np.array(value_u)))

		self.G = G
		time_limit = timelimit
		start = time.time()

		cost = np.inf
		trace = []
		solution = []

		# Step1: get the MST using Prim's
		T_Path = []
		T = np.full_like(self.G, np.inf)
		T, T_Path = self.prim(0)

		#Go through each case of selecting a vertex as the root node
		for i in range(len(city)):
		    # Step2: traverse the MST in preorder walk
		    T_walk = []
		    T_walk = self.preorder_tree_walk(T, i)
		    
		    # Step3: get Hamiltonian cycle
		    H = np.full_like(G, np.inf)
		    H_Path = []
		    H, H_Path, cur_cost = self.create_H(T_walk, i)
		    
		    # Found an improvement, update 
		    if cur_cost < cost:
		        cost = cur_cost
		        trace_time = (time.time() - start)
		        solution = T_walk 
		        # update trace file information
		        trace.append((trace_time, cost))
		        
		    if (time.time() - start) > time_limit:
		        print("Timeout break!!!!")
		        break

		self.cost = cost
		self.solution = solution
		self.trace = trace  

	# compute MST using Prim
	def prim(self, root_index):
	    visited_ids = [root_index] # initialize the set of visited nodes
	    T_Path = []
	    while len(visited_ids) != len(self.G[0]):
	        no_visited_ids = self.contains_no_visited_ids(visited_ids) # maintain the set of non-visited nodes
	        (min_from, min_to), min_weight = self.find_min_edge(visited_ids, no_visited_ids)
	        visited_ids.append(min_to) # maintain the set of visited nodes
	        T_Path.append((min_from, min_to))
	    T = np.full_like(self.G, np.inf) # the matrix form of MST, consisting (n-1) edges
	    for (from_, to_) in T_Path:
	        T[from_][to_] = self.G[from_][to_]
	        T[to_][from_] = self.G[to_][from_]
	    return T, T_Path

	# maintain non-visited nodes set
	def contains_no_visited_ids(self, visited_ids):
	    no_visited_ids = []
	    [no_visited_ids.append(idx) for idx, _ in enumerate(self.G) if idx not in visited_ids]
	    return no_visited_ids

	# Find cheapest edge, add to visited nodes (Greedy)
	def find_min_edge(self, visited_ids, no_visited_ids):
	    min_weight, min_from, min_to = np.inf, np.inf, np.inf
	    for from_index in visited_ids:
	        for to_index, weight in enumerate(self.G[from_index]):
	            if from_index != to_index and weight < min_weight and to_index in no_visited_ids:
	                min_to = to_index
	                min_from = from_index
	                min_weight = self.G[min_from][min_to]
	    return (min_from, min_to), min_weight

	# Preorder the MST
	def preorder_tree_walk(self, T, root_index):
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
	def create_H(self, L, root_index):
	    cost = 0
	    H = np.full_like(self.G, np.inf)
	    H_Path = []
	    for i, from_node in enumerate(L[0:-1]):
	        to_node = L[i + 1]
	        H[from_node][to_node] = self.G[from_node][to_node]
	        H[to_node][from_node] = self.G[to_node][from_node]
	        H_Path.append((from_node, to_node))
	        cost = cost + self.G[from_node][to_node]
	    H_Path.append((to_node, root_index)) # add one last edge to form a cycle
	    cost = cost + self.G[to_node][root_index]
	    return H, H_Path, int(cost)

	def generate_tour(self):
		return self.cost, self.solution, self.trace
