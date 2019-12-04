#CSE6140 project LS1 implemented by Xiaoting Lai
import networkx as nx
import time
import sys
import math
import random

class Opt2Exchange:

    def __init__(self,graph,instance,seed,cutoff=600):
        self.instance = instance
        self.seed = seed
        self.cutoff = cutoff
        self.graph = graph
        self.edges = []
        self.results = []

    # calculate the length between any 2 vertices
    def getEdges(self):
        for i in range(len(self.graph.node.keys())):
            row = []
            for j in range(len(self.graph.node.keys())):
                if i != j:
                    length = int(self.graph[i+1][j+1]['weight'])
                else:
                    length = 0
                row.append(length)
            self.edges.append(row)

    # Calculate the total length of selected routed
    def getRouteLength(self,route):
        length = 0
        for i in range(len(route)):
            u = route[i]
            if i == len(route)-1:
                v = route[0]
            else:
                v = route[i+1]
            length += self.edges[u][v]
        return length

    # switch the order of index u and v
    def exchange2opt(self,route, u, v):
        curr = []
        for i in range(u+1):
            curr.append(route[i])
        for i in range(v,u,-1):
            curr.append(route[i])
        for i in range(v+1, len(route)):
            curr.append(route[i])
        return curr

    # implement 2opt
    def twoOpt(self,start_time):
        end_time = start_time + float(self.cutoff)
        # initialize route
        random.seed(self.seed) 
        curr_route = random.sample(range(len(self.edges)),len(self.edges)) 

        flag = 1 # flag for improvement
        local_min = sys.maxsize
        global_min = sys.maxsize
        global_route = []
        mili = 1.0 / 1000
        # ofile = open(trace_file, 'w')
        while end_time - time.time()  > mili:

            if flag == 0 : # improvement is done for local min
                if local_min < global_min:
                    global_min = local_min
                    global_route = curr_route
                curr_route = random.sample(range(len(self.edges)),len(self.edges))

            local_min = self.getRouteLength(curr_route)
            for u in range(len(curr_route)-1):
                for v in range(u+1, len(curr_route)):
                    flag = 0
                    new_route = self.exchange2opt(curr_route, u, v)
                    curr_length = self.getRouteLength(new_route)
                    if curr_length < local_min:
                        local_min = curr_length
                        curr_route = new_route
                        flag = 1
                        if curr_length < global_min:
                            timestamp = time.time()-start_time
                            self.results.append((curr_route, curr_length, timestamp))
                        break
                if flag == 1:
                    break

        if global_min == sys.maxsize:
            return curr_route 
        else:
            return global_route


    def generate_tour(self):
        start_time = time.time()
        self.getEdges() 
        route = self.twoOpt(start_time)
        total = self.getRouteLength(route)

        return self.results
