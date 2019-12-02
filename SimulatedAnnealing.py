"""
Local Search 2 : Simulated simulatedAnnealing

"""

import time
import sys
import math
import numpy
import random


class SimulatedAnnealing:

    def __init__(self, instance, seed, limit):
        self.city = instance
        self.seed = seed
        self.time_limit = limit
        self.edges = []
        self.graph = {}
        self.result = []

    # helper functions
    # parse input file to store graph in a dictionary
    # format is G{1:[x1, y1], 2:[x2, y2]...n:[xn,yn]}
    def getEdges(self):
        file_path = './DATA/'+ str(self.city) + '.tsp'
        file = open(file_path, 'r')

        description = ''
        for i in range(1, 6):
            description = description + file.readline()

        for line in file:
            nums = line.split()
            if len(nums) == 1:
                break
            else:
                self.graph[int(nums[0])] = [float(nums[1]), float(nums[2])]
        file.close()
        # create edges to record weights between all cities
        for i in range(1, len(self.graph)+1):
            row = []
            for j in range(1, len(self.graph)+1):
                distance = int(math.sqrt((self.graph[i][0] - self.graph[j][0]) ** 2 + (self.graph[i][1] - self.graph[j][1]) ** 2)+0.5)
                row.append(distance)
            self.edges.append(row)


    # Calculate the total distance in routes
    def getRouteLength(self, route):
        length = 0
        for i in range(len(route)):
            u = route[i]
            if i == len(route)-1:
                v = route[0]
            else:
                v = route[i+1]
            length += self.edges[u-1][v-1]
        return length

    # Implementation of Simulated Annealing
    def simulatedAnnealing(self, start_time):
        end_time = int(start_time) + int(self.time_limit)
        random.seed(self.seed)
        # initialize to be an arbitrary existing route
        current_route = random.sample(range(1,len(self.graph)+1),len(self.graph))
        # output = open(output_trace_file, 'w')

        temperature = 1000
        temperature_min = 0.0001
        cooling_rate = 0.99
        best_route = []
        best_distance = 1000000000000
        avg_running_time = 0
        times = 0
        pre_timestamp = start_time
        q = []
        iter = 0
        eps = 1.0 / 100000
        while temperature > temperature_min or end_time - time.time()  > eps:
            # if current route has not been changed for a while, initialize it to an arbitrary route
            flag = all(x == q[0] for x in q)
            if len(q) == 500 and flag:
                # initialize to be arbitrary existing route
                current_route = random.sample(range(1,len(self.graph)+1),len(self.graph))

            # randomly exchange the order of two cities for new route
            index = random.sample(range(len(self.graph)), 2)
            new_route = current_route[:]
            new_route[index[0]], new_route[index[1]] = new_route[index[1]], new_route[index[0]]

            # compare new length with current length
            current_distance = self.getRouteLength(current_route)
            new_distance = self.getRouteLength(new_route)
            diff = new_distance - current_distance

            # keep the new distance definitely if it is shorter than the current one
            # keep the new distance with a probability if it is longer thant the current one
            if diff < 0 or math.exp(-diff/temperature) > random.random():
                current_distance = new_distance
                current_route = new_route[:]

            # record the previous 500 current length in a queue
            if len(q) < 500:
                q.append(current_distance)
            else:
                q.pop(0)
                q.append(current_distance)

            # update improved solution
            if current_distance < best_distance:
                best_distance = current_distance
                best_route = current_route[:]
                # output.write(str(time.time()-start_time) + "  ")
                # output.write(str(best_distance)+"\n")
                self.result.append((best_route, best_distance, (time.time()-start_time)))
                avg_running_time += time.time()-pre_timestamp
                pre_timestamp = time.time()
                times += 1

            # update the temperature at every iteration with slowly cooling down
            temperature = temperature * cooling_rate
            iter += 1
        return best_route

    def generate_tour(self):
        # output_file = './output/' + str(self.city) + "_LS2_" + str(self.time_limit) + "_" + str(self.seed) + ".sol"
        # output_trace_file = './output/' + str(self.city) + "_LS2_" + str(self.time_limit) + "_" + str(self.seed) + ".trace"

        start_time = time.time()
        self.getEdges()
        route = self.simulatedAnnealing(start_time)
        length = self.getRouteLength(route)

        # write optimal route into an output file
        # output = open(output_file, 'w')
        # output.write(str(int(length))+ "\n")
        # for i in range(len(route)):
        #     u = route[i]
        #     if i == len(route)-1:
        #         v = route[0]
        #     else:
        #         v = route[i+1]
        #     edge = [u, v]
        #     weight = self.getRouteLength(edge)/2
        #     output.write(str(u) + " " + str(v) + " " + str(int(weight)) + "\n")
        return self.result
