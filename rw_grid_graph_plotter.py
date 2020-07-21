import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.exception import NetworkXError
from scipy.stats import skew as sk

def graph(timeMin, timeMax, dims, seed):
    """

    :param timeMin: min time step represented on x-axis >1
    :param timeMax: max time step represented on x-axis
    :param dims: tuple where dim[0] is row, dim[1] is col 
    :param seed: the random number seed for np.random.seed

    :return: graphs of mean path length, cluster coef, and skewness

    """
    # clarifies the dimensions
    rows = dims[0]
    cols = dims[1]
    numNodes = rows*cols
    
    # sets the random seed
    np.random.seed(seed)

    # generating agent IDs
    agent_IDs = np.arange(1, numNodes+1)
    
    # generating the coordinates for location stored as tuples
    # this uses something similar to an x y coordinate, so it may seem backwards
    coordinates = [(i, j) for i in range(cols) for j in range(rows)]
    
    # create a list to store cluster coefficients
    cluster_coef = []
    
    # create a list to store mean path length
    mean_path = []
    
    # skew array
    degree_skew = []

    # time steps
    time = np.arange(timeMin, timeMax+1)
    
    # create network 
    network = nx.grid_2d_graph(rows, cols, True)
    
    # this will connect the location with the agent ID
    # happens in order of appearance here according to random agent ID generator
    # this will not change -- physical location
    location_dict = dict(zip(agent_IDs, coordinates))
    nx.draw(network, with_labels=True)
    
    # create a dictionary that will store the current location of each agent
    # initially, it is the exact same as location_dict
    current_loc_dict = dict(zip(agent_IDs, coordinates))
    
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
                    new_loc = (dim-1, y_coord)
                
             # moving east
            if direction == 2:
                new_loc = (x_coord + 1, y_coord)
                
                # check if "moving" off of the board
                if new_loc[0] > dim-1:
                    new_loc = (0, y_coord)
                
            # moving north   
            if direction == 3:
                new_loc = (x_coord, y_coord+1)
                
                if new_loc[1] > dim-1:
                    new_loc = (x_coord, 0)
                    
            # moving south
            if direction == 4:
                new_loc = (x_coord, y_coord-1)
                
                if new_loc[1] < 0:
                    new_loc = (x_coord, dim-1)
                    
                    
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
            
            # if the location is currently empty, don't add anything and move to next loc
            if current_agents != None:
                # I have to make this a dictionary
                current_agents_dict = dict(zip(current_agents, [{}]*len(current_agents)))
            
                # to avoid using a nested for-loop, just grab the first agent at spot
                first_agent = current_agents[0]
                
                # if the agent has made no previous connections, add in the current 
                # connections as a list type
                if network_dict[first_agent] == None:
                    temp = {first_agent: current_agents_dict}
                    network_dict.update(temp)
                    
                # otherwise, extend the current list of connections by adding in 
                # the connections that were just made to first agent
                else:
                    network_dict[first_agent].update(current_agents_dict)
    
                    
                # this will only add the connections to one node
                # meaning the adjacency list won't be symmetric
                # because our graph is undirected this is okay
                # it also has each node "connected" to itself
                # we need this for networkx so that all nodes are accounted for
                
        # networkX cannot allow for a None object in the dictionary
        for node in network_dict:
            if network_dict[node] == None:
                network_dict[node] = {}
                    
        # creates a graph object from a dict of dicts
        graph = nx.convert.from_dict_of_dicts(network_dict)
        
        
        # add mean path length -- issues here
        try:
            # a NetworkXError will be thrown here if there are any nodes with no connections
            tmp = nx.average_shortest_path_length(graph)
            mean_path.append(tmp)
            
        except NetworkXError:
            mean_path.append(None)
        
        # add cluster coef
        cluster_coef.append(nx.average_clustering(graph))
        
        # add average degree
        degrees = list(dict(graph.degree()).values())
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

    


      