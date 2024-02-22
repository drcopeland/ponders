
# -*- coding: utf-8 -*-
"""
PonderThis April 2023
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

TEST = """0011
1101
0110
0001"""

START24 = """000001000000000001110011
110100010110101000010011
011101110000001101001110
000110111000110101101100
101101011010010011101010
111000100101110100101000
110001011100000000000101
100000010001100000000010
000110010010110110101001
011101101011111011100000
011000101010111011111100
100011110010000100100111
000111010010100010001110
011001010001001111110101
110001000010111000100000
000000101100101000101001
111001010010010011110110
100000110001111111011010
110100000011100100110010
101000110111001110010000
110000000010011100100101
111111011011111100010101
000000000110101011100000
110001111100000011001111"""

START30 = """110001000000100101110011001000
010100001100011101010111100110
000110011010011111100010100010
111101110110011101110100110001
000110001000100011001101100010
101111001110110010111101001111
001110000101101001101000001101
111001110000101011111111110100
110000000000110111111001100100
111001110100111110001110111011
111010100010010100000001101100
010111110011001111110100001001
010100111011000001100000011010
010001010110111100100111001101
111111010001011100101100110110
101000110110010111111011001001
111011000100101111101001100010
101001100011010100010000100001
111111100111111110010111110010
010000010000011001001010010011
111010110011011111101100110110
011100110001101001100000000110
111110100101010000100011011010
111100011111000011110001001111
111000111111101011111011100100
101011000011001110101011000011
001101011101000001100101101001
010010100000011011100101010001
010111101001110100010110010010
110000011010111110100110000010"""

def make_grid(raw_data):
    dat = raw_data.split('\n')
    n = len(dat[0])
    grid = np.array([[bool(int(dat[i][j])) for j in range(n)] for i in range(n)])
    return grid

def make_A_matrix(n):
    N = n**2
    A = np.array([[1 if ((int(i/n) == int(j/n)) or (i%n == j%n)) else 0 for j in range(N)] for i in range(N)])
    return A

def unordered_solve(grid):
    # Takes the grid and outputs list of (unordered) coordinates for a solution
    n = grid.shape[0]
    N = n**2
    A = make_A_matrix(n)
    b = np.array([(1 + grid[int(i/n)][i%n]) %2 for i in range(N)])
    x = np.matmul(A,b)%2
    
    unordered_sols = [(int(i/n), i%n) for i in range(N) if x[i]==1]
    
    return unordered_sols


G = nx.Graph()
sols = unordered_solve(make_grid(TEST))

def make_graph(unordered_sols):
    G = nx.Graph()
    G.add_nodes_from(unordered_sols)
    
    for x in G.nodes:
        for y in G.nodes:
            if y == x:
                continue
            if (x[0] == y[0]):
                G.add_edge(x,y, toggle = 'r')
            elif (x[1] == y[1]):
                G.add_edge(x, y, toggle = 'c')
                
    return(G)

grid24 = make_grid(START24)
grid30 = make_grid(START30)

sols24 = unordered_solve(grid24)
sols30 = unordered_solve(grid30)

Gtest = make_graph(sols)
G24 = make_graph(sols24)
G30 = make_graph(sols30)


def make_ij_mat(n, i, j):
    arr = np.array([[1 if a ==i or b == j else 0 for b in range(n)] for a in range(n)])
    return arr


def verify_grid(sols, grid):
    n = grid.shape[0]
    bad_moves = []
    bad_grids = []
    for i, coord in enumerate(sols):
        if grid[coord[0]][coord[1]]:
            print(f'Error turning on bulb at time {i} at coord {coord}')
            bad_moves.append(i)
            bad_grids.append(grid)
            #print(grid)
        grid = (grid + make_ij_mat(n, coord[0], coord[1]))%2
    return (bad_moves, bad_grids) 
    
def backward_verify_grid(sols, grid):
    n = grid.shape[0]
    bad_moves = []
    bad_grids = []
    for i, coord in enumerate(sols):
        if not grid[coord[0]][coord[1]]:
            print(f'Error turning on bulb at time {i} at coord {coord}')
            bad_moves.append(i)
            bad_grids.append(grid)
            #print(grid)
        grid = (grid + make_ij_mat(n, coord[0], coord[1]))%2
    return (bad_moves, bad_grids) 

def greedy_solve1(u_sols, grid):
    n = grid.shape[0]
    remaining_sols = u_sols.copy()
    final_sols = []
    current_grid = grid.copy()
    
    def reduction_step(remaining_sols, current_grid):
        G = make_graph(remaining_sols)
        Hnodes = [node for node in G.nodes if current_grid[node[0]][node[1]]]
        if len(Hnodes) == 0:
            print(f'# Remaining sols: {len(remaining_sols)}')
            return (final_sols, remaining_sols, current_grid)
        H = G.subgraph(Hnodes)
        next_sols = nx.maximal_independent_set(H)
        
        grid_change = sum([make_ij_mat(n, node[0], node[1]) for node in next_sols])%2
        new_grid = (current_grid + grid_change)%2
        
        return (next_sols, new_grid)
    
    found_sol = False
    
    while not found_sol:
        F = reduction_step(remaining_sols, current_grid)
        try:
            next_sols, next_grid = F
        except:
            return F
        final_sols += next_sols
        current_grid = next_grid
        remaining_sols = [sol for sol in remaining_sols if sol not in next_sols]
        if len(remaining_sols) == 0:
            found_sol = True
    
    return final_sols

best_so_far = []
sol = []

sol24 = []
best_so_far24 = []

def trials(u_sols, grid):
    for i in range(100):
        F = greedy_solve1(u_sols, grid)
        try:
            rem_len = len(F[1])
            if rem_len <= 2:
                best_so_far24.append(F)
            else:
                continue
        except:
            sol24.append(F)

def convert_coords(sols, n):
    return [(1 + coord[1], n - coord[0]) for coord in sols]
    
ones30 = np.array([[1 for i in range(30)] for j in range(30)])
ones24 = np.array([[1 for i in range(24)] for j in range(24)]) 