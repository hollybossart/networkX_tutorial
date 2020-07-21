import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import product
from networkx.exception import NetworkXError
from scipy.stats import skew as sk

def graph(timeMin, timeMax, dims, seed):
    """

    :param timeMin: min time step represented on x-axis >1
    :param timeMax: max time step represented on x-axis
    :param dims: tuple where dim[0] is num rows, dim[1] is num cols 
    :param seed: the random number seed for np.random.seed

    :return: graphs of mean path length, cluster coef, and skewness

    """
    # clarifies the dimensions
    rows = dims[0]
    cols = dims[1]

    
    # sets the random seed
    np.random.seed(seed)
    
    # generating the coordinates for location stored as tuples
    # this uses something similar to an x y coordinate, so it may seem backwards
    coordinates = [(i, j) for i in range(cols+1) for j in range(rows+1)]
    
    # create a list to store cluster coefficients
    cluster_coef = []
    
    # create a list to store mean path length
    mean_path = []
    
    # skew array
    degree_skew = []

    # time steps
    time = np.arange(timeMin, timeMax+1)
        
    # make agent ID list
    # agent_IDS are stored as strings even though they are just integers
    # the reason for this is to avoid python getting rid of the leading zero
    agent_IDS = [str(item[0]) + str(item[1]) for item in coordinates]
    
    # create a dictionary that will store the current location of each agent
    # this will eventually change but it starts with each agent at their coordinate
    current_loc_dict = dict(zip(agent_IDS, coordinates))
    
    # create grid network 
    # periodic = True indicates border nodes will wrap around
    network = nx.grid_2d_graph(rows, cols, periodic=True)
    network = nx.relabel_nodes(network, dict(zip(coordinates, agent_IDS)))
    grid = nx.grid_2d_graph(rows, cols, periodic=True)
    grid = nx.relabel_nodes(grid, dict(zip(coordinates, agent_IDS)))
    plt.figure()
    nx.draw_networkx(network)
    plt.show()
    
    
    # looping through time steps
    for i in time:
        
        # making a location dictionary to see who is at each location at any given time
        # this is organized such that each space is a key and the agents on it are the values
        # this is reset at the beginning of each time step
        coord_dict = dict(zip(coordinates, [None]*len(coordinates)))
        
        # looping through each agent to move
        for agent in current_loc_dict.keys():
            
            # randomly decides where the agent will "move" to make connection
            direction = np.random.randint(1,5)
        
            # getting current location of agent
            current_loc = current_loc_dict[agent]
            x_coord = current_loc[0]
            y_coord = current_loc[1]
            
            # moving west
            if direction == 1:
                new_loc = (x_coord - 1, y_coord)
                
                # check if "moving" off of the board
                if new_loc[0] < 0:
                    new_loc = (cols-1, y_coord)
                
             # moving east
            if direction == 2:
                new_loc = (x_coord + 1, y_coord)
                
                # check if "moving" off of the board
                if new_loc[0] > cols-1:
                    new_loc = (0, y_coord)
                
            # moving north   
            if direction == 3:
                new_loc = (x_coord, y_coord+1)
                
                if new_loc[1] > rows-1:
                    new_loc = (x_coord, 0)
                    
            # moving south
            if direction == 4:
                new_loc = (x_coord, y_coord-1)
                
                if new_loc[1] < 0:
                    new_loc = (x_coord, rows-1)
                    
                    
            # now that the agent has "moved" we will add each agent to the dict
            # of who is at each coordinate to establish connections
            # if the current value is None (loc empty), then we need to replace with agent
            if coord_dict[new_loc] == None:
                temp = {new_loc: [agent]}
                coord_dict.update(temp)
            else: 
                # we need to append the current agent on to already existing agents at loc
                coord_dict[new_loc].append(agent)
                
                
            # we also need to save the current location of each agent for further walking
            current_loc_dict[agent] = new_loc
        
        # end of inner for loop -- all agents have moved after this time step 
         
        # now that we know who has made a connection with who, we need to mark that connection 
        for spot in coord_dict:
            
            # current agents at the given coordinate location
            current_agents = coord_dict[spot]
            
            if current_agents != None:
                # creates all possible edge combinations 
                edge_tuples = list(product(current_agents, current_agents))
                temporary = network.edges
                network.add_edges_from(edge_tuples)
        
        network.add_edges_from(grid.edges)
        
        plt.figure()
        nx.draw_networkx(network)
        plt.show()
    
        # add mean path length -- issues here
        try:
            tmp = nx.average_shortest_path_length(network)
            mean_path.append(tmp)
            
        except NetworkXError:
            mean_path.append(None)
        
        # add cluster coef
        cluster_coef.append(nx.average_clustering(network))
        
        # add average degree
        degrees = list(dict(network.degree()).values())
        degree_skew.append(sk(degrees))



    
    # end of time steps
    
    # plot the graph
    # plotting the degree distribution skewness
    plt.figure()
    plt.title('Degree distribution skewness')
    plt.xlabel('Time steps')
    plt.plot(time, degree_skew, color='blue')
    
    
    # plotting the average clustering coefficient
    plt.figure()
    plt.title('Average clustering coefficient (per node for each graph at a time step)')
    plt.xlabel('Time step')
    plt.plot(time, cluster_coef, color='red')
    
    # plotting the mean path length
    plt.figure()
    plt.title('Mean path length')
    plt.xlabel('Time step')
    plt.plot(time, mean_path, color='orange')
    

        

graph(1, 100, (4,3), 5)

    


      