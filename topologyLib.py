from matplotlib import pyplot as plt
import numpy as np
import networkx as nx
import math

def adj2pers(adj):
    # Assume networks are connected
    # G = nx.from_numpy_matrix(adj) MOdified 
    G = nx.from_numpy_array(adj)
    T = nx.maximum_spanning_tree(G)
    MSTedges = T.edges(data=True) # compute maximum spanning tree (MST)
    ccs = sorted([cc[2]['weight'] for cc in MSTedges], reverse=True) # sort all weights in MST
    G.remove_edges_from(MSTedges) # remove MST from the original graph
    nonMSTedges = G.edges(data=True) # find the remaining edges (nonMST) as cycles
    cycles = sorted([cycle[2]['weight'] for cycle in nonMSTedges], reverse=True) # sort all weights in nonMST
    numTotalEdges = (len(adj) * (len(adj) - 1)) / 2
    numZeroEdges = numTotalEdges - len(ccs) - len(cycles)
    if numZeroEdges != 0:
        cycles.extend(int(numZeroEdges) * [0]) # extend 0-valued edges
    return ccs, cycles

def pers2vec(ccs, cycles, numSampledCCs, numSampledCycles):
    # sorted births of ccs as a feature vector
    numCCs = len(ccs)
    ccVec = [math.ceil((i+1)*numCCs/numSampledCCs)-1 for i in range(numSampledCCs)]

    # sorted deaths of cycles as a feature vector
    numCycless = len(cycles)
    cycleVec = [math.ceil((i+1)*numCycless/numSampledCycles)-1 for i in range(numSampledCycles)]

    # return ccVec + cycleVec # concatenate into a unified feature vector

    npCCs = np.array(ccs)
    npCycles = np.array(cycles)
    return list(npCCs[ccVec]) + list(npCycles[cycleVec])

def pers2vecNodes(ccs, cycles, numNodes):

    # Introduce a hyperpaprameter N as # of nodes in a graph, such that N - 1 is # of components and (N choose 2) - (N - 1) is # of cycles.

    # Calculate numSampledCCs and numSampledCycles
    numSampledCCs = numNodes - 1
    numSampledCycles = n_choose_2(numNodes) - (numNodes - 1)

    # sorted births of ccs as a feature vector
    numCCs = len(ccs)
    ccVec = [math.ceil((i+1)*numCCs/numSampledCCs)-1 for i in range(numSampledCCs)]

    # sorted deaths of cycles as a feature vector
    numCycless = len(cycles)
    cycleVec = [math.ceil((i+1)*numCycless/numSampledCycles)-1 for i in range(numSampledCycles)]

    # return ccVec + cycleVec # concatenate into a unified feature vector

    npCCs = np.array(ccs)
    npCycles = np.array(cycles)
    return list(npCCs[ccVec]) + list(npCycles[cycleVec])

def n_choose_2(n):
    return n * (n - 1) // 2