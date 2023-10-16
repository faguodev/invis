'''
Created on Jan 6, 2014

@author: doglic
'''

import numpy as np
import comparable as comp
import warnings


class graph(object):

    def __init__(self, n):
        self.__n = n 
        self.__m = 0
        self.__adjs = np.empty(n, dtype=object)
        for i in range(n):
            self.__adjs[i] = []
    
    
    def destruct(self):
        warnings.warn('graph object is being destroyed...')
        i = 0
        j = 0
        while i < self.__n:
            adj = self.__adjs[0]
            m = len(adj)
            while j < m:
                if isinstance(adj[0], edge):
                    adj[0].destruct()
                del adj[0]
                j += 1
            del adj
            i += 1
        del self.__adjs
        del self.__n
        del self.__m
    
    
    def v(self):
        return self.__n
    
    
    def e(self):
        return self.__m
    
    
    def add_edge(self, u, v, w):
        if u < 0 or u >= self.__n or v < 0 or v >= self.__n:
            raise IndexError('Invalid vertex! Vertex index must be greater than 0 and less than ' + str(self.__n))
        self.__adjs[u].append(edge(u, v, w))
        self.__adjs[v].append(edge(v, u, w))
        self.__m += 1
    
    
    def adj(self, u):
        if u < 0 or u >= self.__n:
            raise IndexError('Invalid vertex! Vertex index must be greater than 0 and less than ' + str(self.__n))
        return self.__adjs[u]


    def edges(self):
        unique_edges = []
        for i in range(self.__n):
            stored_self_loops = 0
            for i_edge in self.__adjs[i]:
                if i_edge.head() > i:
                    unique_edges.append(i_edge)
                elif i_edge.head() == i and stored_self_loops % 2 == 0:
                    unique_edges.append(i_edge)
                    stored_self_loops += 1
        
        return unique_edges
    

class edge(comp.comparable):
    
    
    def __init__(self, u, v, w):
        self.__u = u
        self.__v = v
        
        if None == w:
            raise ValueError('Edge weight not defined!')
        else:
            self.__w = w
    
    
    def destruct(self):
        del self.__u
        del self.__v
        del self.__w
    
    
    def weight(self):
        return self.__w
    
    
    def head(self):
        return self.__v
    
    
    def tail(self):
        return self.__u


    def _cmp_key(self):
        return self.__w