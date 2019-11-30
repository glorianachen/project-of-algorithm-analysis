#CSE6140 project LS1 implemented by Xiaoting Lai
import time
import sys
import math
import numpy
import random


class Opt2Exchange:

    def __init__(self,instance,seed,cutoff=600):
        self.instance = instance
        self.seed = seed
        self.cutoff = cutoff
        self.graph = {}
        self.edges = []

    # calculate the edges between any 2 vertices
    def getEdges(self):
        file_path = './DATA/'+ self.instance + '.tsp'

        file = open(file_path, 'r')
        for i in range(0,6):
            line = file.readline()
        while line!='EOF\n':
            nums = line.split()
            self.graph[int(nums[0])-1] = [float(nums[1]), float(nums[2])]
            line=file.readline()
        file.close()

        for i in range(len(self.graph)):
            row = []
            for j in range(len(self.graph)):
                length = round(math.sqrt((self.graph[i][0] - self.graph[j][0])**2 + (self.graph[i][1] - self.graph[j][1])**2))
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
            length += self.edges[u-1][v-1]
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
    def twoOpt(self, trace_file, start_time):
        end_time = start_time + float(self.cutoff)
        # initialize route
        random.seed(self.seed) 
        curr_route = random.sample(range(len(self.edges)),len(self.edges)) 

        flag = 1 # flag for improvement
        local_min = sys.maxsize
        global_min = sys.maxsize
        global_route = []
        mili = 1.0 / 1000
        ofile = open(trace_file, 'w')
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
                            timestamp = (time.time()-start_time)*1000
                            ofile.write(f"{timestamp:.2f}, ") # printed unit is ms
                            ofile.write(str(curr_length)+"\n")
                        break
                if flag == 1:
                    break

        if global_min == sys.maxsize:
            return curr_route 
        else:
            return global_route


    def generate_tour(self):
        start_time = time.time()
        sol_file = './Output/' + self.instance + "_LS1_" + str(self.cutoff) + "_" + str(self.seed) + ".sol"
        trace_file = './Output/' + self.instance + "_LS1_" + str(self.cutoff) + "_" + str(self.seed) + ".trace"

        self.getEdges() 
        route = self.twoOpt(trace_file, start_time)
        total = self.getRouteLength(route)

        ofile = open(sol_file, 'w')
        ofile.write(str(total) + "\n")
        for i in range(len(route)-1):
            ofile.write(str(route[i])+",")
        ofile.write(str(route[len(route)-1]))