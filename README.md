# Search algorithms for 8-Puzzle game

This is a simple implementation for 8-Puzzle game solver.
To execute: 
```
python driver.py <algorithm> <board configuration>
```
For example, to execute A* for the board:

| 6 | 1 | 8 |
| - | - | - |
| 4 | 0 | 2 |
| 7 | 3 | 5 |

Execute:

```
python driver.py ast 6,1,8,4,0,2,7,3,5
```

## Algorithms

Implemented algorithms:

### Breadth First Search

```
python driver.py bfs 6,1,8,4,0,2,7,3,5
```

### Depth First Search

```
python driver.py dfs 6,1,8,4,0,2,7,3,5
```

### A* Search


```
python driver.py ast 6,1,8,4,0,2,7,3,5
```
