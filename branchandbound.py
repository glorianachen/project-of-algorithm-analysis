#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 17:43:11 2019

@author: chenjiayu
"""

# Class which implements the branch and bound algorithm using the NodeStatus.py file, 
# we use DFS for out branch and bound implentation, by Bhanu Verma
import networkx as nx
import sys
import numpy as np
import time



class NodeStatus:

    def __init__(self,graph, path, path_cost,mat):
        self.graph = graph
        self.path = path
        self.path_cost = path_cost
        self.mat = mat
    
    def get_cheapest_neighbour(self,node,graph,path):

        #node edge dict
        temp_dict = graph[node]
        neighbours= []
        #list of (neighbour, weight)
        for key in temp_dict.keys():
            if key not in path:
                neighbours.append((key,temp_dict[key]['weight']))
        #find neighbour with best weight 
        if neighbours:
            return min(neighbours,key =lambda p:p[1])[0]
        else:
            return 1


    # GLOBAL! path cost update, path update, calculate bound
    def add_new(self, new):
        # only when not empty path, include the stop to path, update the path cost and lb
        if len(self.path):
            self.path_cost = self.path_cost + self.graph[self.path[-1]][new]['weight']
        self.path.append(new)
#2 ways to bound SELF??
        self.bound_val = self.get_lower_bound(self.graph, self.path, self.path_cost)
		#self.bound_val = self.get_min_dist_lower_bound(self.mat, self.path, self.path_cost)
        

    # lowerbound
    def get_lower_bound(self, graph, path, path_cost):
        if len(path)==len(graph.node.keys()):
            return path_cost+graph[path[0]][path[-1]]['weight']
        if len(path)==len(graph.node.keys())+1:
            return path_cost
        sub_graph = graph.copy()
# why??? just in case delete twice ? if path is cycle
        if len(path) > 1 and path[0] == path[-1]:
            # trim last redundant one
            path = path[:-1]
            #maybe trim already chosen ones     
        for n in path:
            sub_graph.remove_node(n)
            # get mst for trimmed graph
        mst = nx.minimum_spanning_tree(sub_graph)
        #v is best neighbor for u, extend path on both ends
        endnode1 = path[0]
        endnode1_neighbour = self.get_cheapest_neighbour(endnode1,graph,path)
        endnode2 = path[-1]
        endnode2_neighbour= self.get_cheapest_neighbour(endnode2,graph,path)
        mst.add_edge(endnode1, endnode1_neighbour, weight=graph[endnode1][endnode1_neighbour]['weight'])
        mst.add_edge(endnode2, endnode2_neighbour, weight= graph[endnode2][endnode2_neighbour]['weight'])
        return path_cost + mst.size(weight='weight')
    #reduced mat cost+path
'''
	def get_min_dist_lower_bound(self, mat, path, path_cost):
		mat = mat.copy();size=len(mat) 
		if len(path) > 1:
			i=0
''' 
            #missing infty for columns?
'''
			while i < len(path)-2:
                #set row of path elements infty, why 2 steps?
				mat[path[i]-1] = np.array([sys.maxsize]*size)
                mat.T[path[i]-1] = np.array([sys.maxsize]*size)
				i += 1
        #row min
		row_min = np.amin(mat, axis=1)
        #mat-vector of rowmin
		mat = mat - np.reshape(row_min,(size,1))
        #now column min
		col_min = np.amin(mat, axis=0)
        #reduced mat cost+path
		return np.sum(row_min) + np.sum(col_min) + path_cost
'''

    
    
class BranchAndBound:

    def __init__(self, graph, cutoff=600):
        self.graph = graph
        self.winner = None
        self.results = []
        self.cutoff = cutoff


    def run_DFS(self, graph, mat):
        node_stack = []
        initial_city = 1
        #first_node is a state object 
        first_node = NodeStatus(graph.copy(), [], 0, mat)
        first_node.add_new(initial_city)
        #LIST of obj
        node_stack.append(first_node)
		
        while len(node_stack):
            #running time stop
            if time.time() - self.begin_time > self.cutoff:
                break
            #last node in tree!
            last_node = node_stack.pop()
            
            # if missing winner or last obj could be better than winner, we name it winner
            if not self.winner or last_node.bound_val < self.winner.path_cost:
                
				# checking if we could get a complete path
                if len(graph.node.keys()) == len(last_node.path):

					# checking if we have a cycle
                    if last_node.path[0]!= last_node.path[-1]: 
                        last_node.add_new(last_node.path[0])

						# checking if upper bound needs to be updated, missing first winner or is better than winenr
                    if not self.winner or self.winner.path_cost > last_node.path_cost:
                            # winner is the last one
                        index_change_path=[x-1 for x in last_node.path]
                        self.results.append((index_change_path, last_node.path_cost, time.time() - self.begin_time))
                        self.winner = last_node
                else:
                    sorted_edge = self.sort_edges(graph[last_node.path[-1]])
                    for node,cost in sorted_edge:
                        if node not in last_node.path:
                            new_kid = NodeStatus(graph, last_node.path.copy(), last_node.path_cost, mat)
                            new_kid.add_new(node)

							# only when promising, we keep the new possiblity
                            if not self.winner or self.winner.path_cost > new_kid.bound_val: 
                                node_stack.append(new_kid)

    
    def sort_edges(self,edges):
        temp_kids = []
        for key in edges:
            temp_kids.append((key,edges[key]['weight']))
        # sort in decreasing order of weight
        temp2=sorted(temp_kids, key=lambda p:p[1], reverse=True)
        return temp2

# what if some column is cut???
    def generate_tour(self):
        graph = self.graph
        #generate mat, notice 0 and 1 index!!
        size=len(graph.node.keys())
        mat=np.zeros((size,size))       
		#mat = [[0 for i in range(len(graph.node.keys()))] for j in range(len(graph.node.keys()))]
        for i in range(size):
            for j in range(size):
                if i != j:
                    mat[i][j] = graph[i+1][j+1]['weight']
                else:
                    mat[i][j] = sys.maxsize
        self.mat = mat
        self.begin_time = time.time()
        self.run_DFS(graph, self.mat)

        return self.results
