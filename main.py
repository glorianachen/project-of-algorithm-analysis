import networkx as nx
from os.path import isfile
from math import sqrt
import argparse
import sys
# import branchandbound
import Opt2Exchange

class TSP:
    def __init__(self, graph):
        self.graph = graph

    def generate(self, instance='Cincinnati', algorithm='BnB', cutoff=60,seed=1):
        if algorithm == 'BnB':
            bnb = branchandbound.BranchAndBound(self.graph,cutoff)
            return bnb.generate_tour()
        elif algorithm == 'LS1':
            ls_1 = Opt2Exchange.Opt2Exchange(instance,seed,cutoff)
            ls_1.generate_tour()
        else:
            return None

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
    parser.add_argument('-inst', type=str, dest="instance", default='Cincinnati', 
    help='Input of city name')
    parser.add_argument('-alg', type=str, dest="algorithm", default='LS1', 
    help='Input of chosen algorithm')
    parser.add_argument('-time', type=int, dest="cutoff", default=5, 
    help='Input of cutoff time')
    parser.add_argument('-seed', type=int, dest="seed", default=0, 
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
                val = round(sqrt(delta_x**2 + delta_y**2))
                graph.add_edge(u, v, weight= val)
       
    # sending graph to algorithm
    solver = TSP(graph)
    kwargs = vars(args).copy()
    '''
    Shouldnt we pass results to tour_data?
    '''

    
    #Output file name
    if args.algorithm == 'BnB':
        final_results=solver.generate(**kwargs)
        file_name = str(args.instance) + '_' + str(args.algorithm) + '_' + str(args.cutoff)
        sol_file = file_name + '.sol'
        trace_file = file_name + '.trace'



            #took from others, need to modify in the end

            # finalresults=list of (last_state.path, last_state.path_cost, time.time() - self.begin_time))
             #Generating solution file?????? no weight
        f=open(sol_file, 'w')
        f.write('{}\n'.format(final_results[-1][1]))
        for edge in final_results[-1][:-2]:
            f.write('{},'.format(edge))
        f.write(final_results[-1][-2])

            # Generating trace file
        f=open(trace_file, 'w')
        for entry in final_results:
            f.write('{:.2f}, {}\n'.format(entry[2], entry[1]))
    else:
        solver.generate(**kwargs)
    
            
''''
        if final_results:
            opt = opt_tour_lengths[args.instance]
            rel_err = round(abs(final_results[-1][1] - opt)/opt,4)
            print('Relative error is ', rel_err)
'''

if __name__ == '__main__':
    main()
    
    

        
"""
        elif algorithm == 'MSTApprox':
            approx_1 = MSTApprox.MSTApprox(self.graph,instance,seed,cutoff)
            approx_1.generate_tour()
        elif algorithm == 'LS1':
            ls_1 = Opt2Search.Opt2Search(self.graph,instance,seed,cutoff)
            ls_1.generate_tour()
        elif algorithm == 'LS2':
            ls_2 = SimulatedAnnealing.SimulatedAnnealing(self.graph,instance,seed,cutoff)
            ls_2.generate_tour()
"""
