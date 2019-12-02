import networkx as nx
from os.path import isfile
from math import sqrt
import argparse
import sys
import os
import branchandbound
import Opt2Exchange
import SimulatedAnnealing
import approx


class TSP:
    def __init__(self, graph):
        self.graph = graph

    def generate(self, instance='Cincinnati', algorithm='BnB', cutoff=600,seed=1):
        if algorithm == 'BnB':
            bnb = branchandbound.BranchAndBound(self.graph,cutoff)
            return bnb.generate_tour()
        elif algorithm == 'LS1':
            ls_1 = Opt2Exchange.Opt2Exchange(self.graph,instance,seed,cutoff)
            return ls_1.generate_tour()
        elif algorithm == 'LS2':
            ls_2 = SimulatedAnnealing.SimulatedAnnealing(instance,seed,cutoff)
            return ls_2.generate_tour()
        else:
            return None

def main():

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
    
    ###############  Added by Xueqing, needed for verifying
    # File IO: extract city name from the path form args.instance in order to better use in write file
    # example:  args.instance = DATA/Atlanta.tsp => city_name = Atlanta
    sub = args.instance.rfind('/')
    if sub == -1:
        city_name = args.instance[:-4]
    else:
        city_name = args.instance[(sub+1):-4]
    ############### Added by Xueqing
    
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
        'Denver': 100431,
        'Berlin': 7542
    }
    opt = optimal_tour_lengths[args.instance]

    #Output file name for BnB
    if args.algorithm == 'BnB':
        # final results form: [[1st],[2nd],[3rd]]. example[[[0, 3, 2, 1, 0], 200, 0.5], [[0, 2, 1, 3, 0], 150, 0.7]]
        final_results=solver.generate(**kwargs)
        
        file_name = str(args.instance) + '_' + str(args.algorithm) + '_' + str(args.cutoff)
        sol_file = file_name + '.sol'
        trace_file = file_name + '.trace'

            # finalresults=list of (last_state.path, last_state.path_cost, time.time() - self.begin_time))
        f=open(sol_file, 'w')
        f.write('{}\n'.format(final_results[-1][1]))
        for edge in final_results[-1][0][:-2]:
            f.write('{},'.format(edge))
        f.write(final_results[-1][0][-2])
        f.close()

            # Generating trace file
        f=open(trace_file, 'w')
        for entry in final_results:
            f.write('{:.2f}, {}\n'.format(entry[2], entry[1]))
        f.close()
    
    elif args.algorithm == 'Approx':
        seed = 0
        approx_cost, approx_solution, approx_trace, rel_error = approx.Approx(city, args.cutoff, seed, args.instance).generate_tour()
        #approx_cost, approx_solution, approx_trace = solver.generate(**kwargs)

        if 'output' not in os.listdir('./'):
            os.mkdir('./output')
        #with open('output/'+ city_name + "_Approx_" + str(args.cutoff) +'.trace', 'w') as f:
        with open('output/'+ args.instance + "_Approx_" + str(args.cutoff) +'.trace', 'w') as f:
            for (a, b) in approx_trace:
                f.write('{:.2f}, {}\n'.format(a, b))
        #with open('output/'+ city_name + "_Approx_" + str(args.cutoff) +'.sol', 'w') as f:
        with open('output/'+ args.instance + "_Approx_" + str(args.cutoff) +'.sol', 'w') as f:
            f.write('{}\n'.format(approx_cost))
            for vertex in approx_solution[:-1]:
                f.write(str(vertex) + ',') 
            f.write(str(approx_solution[-1])) 
    
    # write into file with format    
    elif args.algorithm == 'LS1':
        # final results form: [[1st],[2nd],[3rd]]. example[[[0, 3, 2, 1, 0], 200, 0.5], [[0, 2, 1, 3, 0], 150, 0.7]]
        final_results=solver.generate(**kwargs)
        
        sol_file = './output/' + args.instance + "_LS1_" + str(args.cutoff) + "_" + str(args.seed) + ".sol"
        trace_file = './output/' + args.instance + "_LS1_" + str(args.cutoff) + "_" + str(args.seed) + ".trace"

            # finalresults=list of (last_state.path, last_state.path_cost, time.time() - self.begin_time))
        f=open(sol_file, 'w')
        f.write('{}\n'.format(final_results[-1][1]))
        for edge in final_results[-1][0][:-1]:
            f.write('{},'.format(edge))
        f.write(str(final_results[-1][0][-1]))
        f.close()

            # Generating trace file
        f=open(trace_file, 'w')
        for entry in final_results:
            f.write('{:.2f}, {}\n'.format(entry[2], entry[1]))
        f.close()

        error = round(abs(final_results[-1][1] - opt)/opt,4)
        end_time = final_results[-1][2]
        print('Relative error is ', error)
        print('Finished time is {:.2f}'.format(end_time))
        
    elif args.algorithm == 'LS2':
        # final results form: [[1st],[2nd],[3rd]]. example[[[0, 3, 2, 1, 0], 200, 0.5], [[0, 2, 1, 3, 0], 150, 0.7]]
        final_results = solver.generate(**kwargs)

        sol_file = './output/' + args.instance + "_LS2_" + str(args.cutoff) + "_" + str(args.seed) + ".sol"
        trace_file = './output/' + args.instance + "_LS2_" + str(args.cutoff) + "_" + str(args.seed) + ".trace"

        # finalresults=list of (last_state.path, last_state.path_cost, time.time() - self.begin_time))
        f = open(sol_file, 'w')
        f.write('{}\n'.format(final_results[-1][1]))
        for edge in final_results[-1][0][:-1]:
            f.write('{},'.format(edge))
        f.write(str(final_results[-1][0][-1]))
        f.close()

        # Generating trace file
        f = open(trace_file, 'w')
        for entry in final_results:
            f.write('{:.2f}, {}\n'.format(entry[2], entry[1]))
        f.close()

        error = round(abs(final_results[-1][1] - opt)/opt,4)
        end_time = final_results[-1][2]
        print('Relative error is ', error)
        print('Finished time is {:.2f}'.format(end_time))

    else:
        solver.generate(**kwargs)
    

    
"""            
    if final_results:
        error = round(abs(final_results[-1][1] - opt)/opt,4)
        end_time = final_results[-1][2]
        print('Relative error is ', error)
        print('Finished time is {:.2f}'.format(end_time))
"""


if __name__ == '__main__':
    main()
    
    

        
"""
        elif algorithm == 'MSTApprox':
            approx_1 = MSTApprox.MSTApprox(self.graph,instance,seed,cutoff)
            approx_1.generate_tour()
        elif algorithm == 'LS1':
            ls_1 = Opt2Search.Opt2Search(self.graph,instance,seed,cutoff)
            ls_1.generate_tour()

"""
