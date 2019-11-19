import networkx as nx
from os.path import isfile
from math import sqrt
import argparse
import sys
import math
import branchandbound

def main():

    # Optimal 
    optimal_tour_lengths = {
        'SanFrancisco': 810196,
        'NYC': 1555060,
        'Roanoke': 655454,
        'Atlanta': 2003763,
        'Champaign': 52643,
        'Cincinnati': 277952,
        'Philadelphia': 1395981,
        'UKansasState': 62962,
        'Toronto': 1176151,
        'UMissouri': 132709,
        'Boston': 893536,
        'Denver': 100431
    }

    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-inst', type=str, dest="instance", default='Champaign', 
    help='Input of city name')
    parser.add_argument('-alg', type=str, dest="algorithm", default='bnb', 
    help='Input of chosen algorithm')
    parser.add_argument('-time', type=int, dest="cutoff", default=1, 
    help='Input of cutoff time')
    parser.add_argument('-seed', type=int, dest="seed", default=1, 
    help='Input of random seed')
    args = parser.parse_args()
    
    # translate instance into 'xy coordinate graph', notice the main.py need to be in the mother file of DATA
    citymap = "./DATA/{}.tsp".format(args.instance)
    if not isfile(citymap):
        print("File not found. Check the city name!")
        sys.exit()
    f=open(citymap)
    line=f.readline()
    while not 'NODE_COORD_SECTION\n' in line:
      line=f.readline()
    line=f.readline()
    city={}
    while line!='EOF\n':
      v = line.split(' ')
      city[int(v[0])] = [float(v[1]), float(v[2])]
      line=f.readline()
      
    # calculate distance for edges. access edges by e.g. graph[1][2]['weight']
    graph= nx.Graph()
    for u in city:
        for v in city:
            if u != v:
                delta_x = city[u][0] - city[v][0]
                delta_y = city[u][1] - city[v][1]
                val = round(math.sqrt(delta_x**2 + delta_y**2))
                graph.add_edge(u, v, weight= val)
       
    # sending graph to algorithm
    solver = TSP(graph)
    kwargs = vars(args).copy()
    solver.build_tour(**kwargs)
    
    #Output file name
    if args.algorithm == 'BnB':
        file_name = str(args.instance) + '_' + str(args.algorithm) + '_' + str(args.cutoff)
    else:
        file_name = str(args.instance) + '_' + str(args.algorithm) + '_' + str(args.cutoff) + '_' + str(args.seed)
    sol_file = file_name + '.sol'
    trace_file = file_name + '.trace'



    '''' took from others, need to modify in the end
        # Generating solution file?????? no weight
        f=open(sol_file, 'w')
        f.write('{}\n'.format(tour_data[-1][1]))
        for edge in zip(tour_data[-1][0], tour_data[-1][0][1:]):
            f.write('{} {} {}\n'.format(edge[0], edge[1], graph[edge[0]][edge[1]]['weight']))

        # Generating trace file
        f=open(trace_file, 'w')
        for entry in tour_data:
            f.write('{:.2f} {}\n'.format(entry[2], entry[1]))
            
        ????
        if tour_data:
            opt = opt_tour_lengths[args.instance]
            rel_err = (tour_data[-1][1] - opt)/opt
            print('Relative error is ', rel_err)
    ''''

if __name__ == '__main__':
    main()
    
    
    
    
class TSP:
    def __init__(self, graph):
        self.graph = graph

    def build_tour(self, instance='Cincinnati', algorithm='BnB', cutoff=1,seed=1):
        if algorithm == 'BnB':
            bnb = branchandbound.BranchAndBound(self.graph,cutoff)
            return bnb.generate_tour()
        
        ''''
        elif algorithm == 'MSTApprox':
            approx_1 = MSTApprox.MSTApprox(self.graph,instance,seed,cutoff)
            approx_1.generate_tour()
        elif algorithm == 'LS1':
            ls_1 = Opt2Search.Opt2Search(self.graph,instance,seed,cutoff)
            ls_1.generate_tour()
        elif algorithm == 'LS2':
            ls_2 = SimulatedAnnealing.SimulatedAnnealing(self.graph,instance,seed,cutoff)
            ls_2.generate_tour()
        ''''
        else:
            return None
