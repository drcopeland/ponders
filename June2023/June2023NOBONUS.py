# -*- coding: utf-8 -*-
"""
June 2023 Ponder This

"""
import numpy as np

def f(x):
    a = 1103515245
    c = 12345
    m = 2**31
    
    return (a*x + c) % m

def times_from_pos(pos, n):
    times = []
    prod = pos[0]*pos[1]*pos[2]
    for i in range(int(np.ceil(n/2))):
        times.append(f(prod + i) % n)
        
    return set(times)


a = 1103515245
c = 12345
m = 2**31       

def check_sol(sol, k, n):
    cheese = 0
    cheese_pos = []
    
    pos = [1,1,1]
    for t in range(len(sol)):
        if (t+1) % n in times_from_pos(pos, n):
            if pos not in cheese_pos:
                cheese += 1
                cheese_pos.append(pos.copy())
                print(f'Cheese at time {t+1} and position {pos}.')
        if sol[t] == 'R':
            pos[0] += 1
        elif sol[t] == 'U':
            pos[1] += 1
        elif sol[t] == 'F':
            pos[2] += 1
            
    return cheese, cheese_pos

# return a dictionary with keys = possible next moves, values = positions after next move
def next_pos(pos, k):
    next_pos = {'W': pos}
    if pos[0] < k:
        next_pos['R'] = (pos[0] + 1, pos[1], pos[2])
    if pos[1] < k:
        next_pos['U'] = (pos[0], pos[1] + 1, pos[2])
    if pos[2] < k:
        next_pos['F'] = (pos[0], pos[1], pos[2]+1)
    
    return next_pos

# initialize a grid dictionary
# keys = positions
# values = (most cheeses obtainable in T steps, cheesiest path of length T) (initially T = 0)

def init_grid(k, n):
    grid = {}
    for a in range(k):
        for b in range(k):
            for c in range(k):
                pos = (a+1, b+1, c+1)
                if 0 in times_from_pos(pos, n):
                    grid[pos] = (1, '*E') # end paths with E for convenience
                else:
                    grid[pos] = (0, 'E')
    return grid

# Update step for the grid dictionary
def update_grid(k, n, prev_grid, T):
    if T == 0:
        return prev_grid
    else:
        t = n - T
        new_grid = prev_grid.copy()
        for pos in prev_grid.keys():
            times = times_from_pos(pos, n)
            # Case 1: current square has no cheese
            if t not in times:
                best_cheese, best_move, best_pos = 0, '', ()
                
                # Find the optimal next move
                for move, npos in next_pos(pos, k).items():
                    if prev_grid[npos][0] >= best_cheese:
                        best_cheese = prev_grid[npos][0]
                        best_move = move
                        best_pos = npos
                        
                # Update the grid
                if best_move == 'W' and prev_grid[pos][1][0] == '*':
                    new_grid[pos] = (best_cheese, '*W' + prev_grid[pos][1])
                else:
                    new_grid[pos] = (best_cheese, best_move + prev_grid[best_pos][1])
                 
            # Case 2: current square has cheese        
            else:
                best_cheese, best_move, best_pos = 0, '', ()
                
                # Find the optimal next move
                for move, npos in next_pos(pos, k).items():
                    if move == 'W' and prev_grid[npos][1][0] == '*':
                        if prev_grid[npos][0] >= best_cheese:
                            best_cheese = prev_grid[npos][0]
                            best_move = move
                            best_pos = npos
                    else:
                        if prev_grid[npos][0] + 1 >= best_cheese:
                            best_cheese = prev_grid[npos][0] + 1
                            best_move = move
                            best_pos = npos
            
                # Update the grid
                new_grid[pos] = (best_cheese, '*' + best_move + prev_grid[best_pos][1])
                
    return new_grid

def find_sols(k, n):
    grid = init_grid(k, n)
    T = 0
    while T < n:
        grid = update_grid(k, n, grid, T)
        T += 1
        
    cheese = grid[(1,1,1)][0]
    sol = grid[(1,1,1)][1].replace('*','')[:-1]
        
    return cheese, sol
                
import time

start = time.time()
# k, n = 5, 20
k, n = 30, 100
cheese, path = find_sols(k, n) # k=5, n=20
print(cheese, path) 
end = time.time()
print(f'Time: {end - start}')
              
check_sol(path, k, n)                
                
                
                
                