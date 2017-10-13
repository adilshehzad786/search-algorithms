from __future__ import print_function
import sys
import copy
import time
import resource
import collections
import heapq

class PriorityQueue:
    def __init__(self):
        self._queue = []
        self._index = 0

    def push(self, item, priority):
        heapq.heappush(self._queue, (-priority, self._index, item))
        self._index += 1

    def pop(self):
        return heapq.heappop(self._queue)[-1]

def memory_usage_resource():
    rusage_denom = 1024. * 1024.
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / rusage_denom
    return mem

class SearchAlgorithm:
    def __init__(self, board):
        self.board = board
        self.init_frontier()
        self.nodes_expanded = 0
        self.max_search_depth = 0
        self.max_ram_usage = 0
    
    def order_expanded(self, list):
        return list

    def search(self):
        start_time = time.time()
        visited = set()
        frontier_set = set()
        frontier_set.add(self.frontier[0])
        goal = Board.goal()
        nodes_expanded = 0
        while not self.is_frontier_empty():
            state = self.get_next()
            visited.add(state.state)

            self.max_ram_usage = max(self.max_ram_usage, memory_usage_resource())
            if state.state == goal:
                curr_state = state
                path = []
                while curr_state.parent:
                    path.insert(0, curr_state.moviment)
                    curr_state = curr_state.parent
                return {
                    'moviments': path,
                    'nodes_expanded': nodes_expanded,
                    'depth': state.depth,
                    'running_time': time.time() - start_time,
                    'max_ram_usage': self.max_ram_usage
                }

            nodes_expanded += 1
            for expanded in self.order_expanded(state.get_expanded()):
                if expanded.state not in visited and expanded.state not in frontier_set:
                    self.insert(expanded)
                    frontier_set.add(expanded.state)

class BreadthFirstSearch(SearchAlgorithm):
    def init_frontier(self):
        self.frontier = collections.deque([self.board])

    def insert(self, state):
        self.frontier.appendleft(state)

    def get_next(self):
        return self.frontier.pop()

    def is_frontier_empty(self):
        return len(self.frontier) == 0

class DepthFirstSearch(SearchAlgorithm):
    def init_frontier(self):
        self.frontier = collections.deque([self.board])

    def insert(self, state):
        self.frontier.append(state)

    def get_next(self):
        return self.frontier.pop()

    def is_frontier_empty(self):
        return len(self.frontier) == 0

    def order_expanded(self, l):
        _l = list(l)
        _l.reverse()
        return _l


class AStarSearch(SearchAlgorithm):
    def init_frontier(self):
        self.frontier = [self.board]

    def insert(self, state):
        self.frontier.append(state)

    def get_next(self):
        return self.frontier.pop()

    def is_frontier_empty(self):
        return len(self.frontier) == 0

class Board:
    LENGTH = 3
    MAX_DEPTH = 0
    def __init__(self, state, parent = None, cost = 0, moviment = None, depth = 0):
        self.state = state
        self.parent = parent
        self.cost = cost
        self.moviment = moviment
        self.depth = depth
        Board.MAX_DEPTH = max(Board.MAX_DEPTH, depth)
    def __str__(self):
        return self.state

    @staticmethod
    def up(state):
        state_l = list(state)
        index = state_l.index(0)
        if index not in range(Board.LENGTH):
            state_l[index - Board.LENGTH], state_l[index] = state_l[index], state_l[index - Board.LENGTH]
            state = tuple(state_l)
            return state
        else:
            return 0

    @staticmethod
    def down(state):
        state_l = list(state)
        index = state_l.index(0)
        if index not in range(Board.LENGTH * (Board.LENGTH - 1),Board.LENGTH * Board.LENGTH):
            state_l[index + Board.LENGTH], state_l[index] = state_l[index], state_l[index + Board.LENGTH]
            state = tuple(state_l)
            return state
        else:
            return 0

    @staticmethod
    def left(state):
        state_l = list(state)
        index = state_l.index(0)
        if index not in range(0,len(state), Board.LENGTH):
            state_l[index-1], state_l[index] = state_l[index], state_l[index-1]
            state = tuple(state_l)
            return state
        else:
            return 0

    @staticmethod
    def right(state):
        state_l = list(state)
        index = state_l.index(0)
        if index not in range(Board.LENGTH - 1,len(state_l), Board.LENGTH):
            state_l[index + 1], state_l[index] = state_l[index], state_l[index + 1]
            state = tuple(state_l)
            return state
        else:
            return 0

    def get_expanded(self):
        expanded = []
        expanded.append(Board(Board.up(self.state), self, self.cost + 1, 'Up', self.depth + 1))
        expanded.append(Board(Board.down(self.state), self, self.cost + 1, 'Down', self.depth + 1))
        expanded.append(Board(Board.left(self.state), self, self.cost + 1, 'Left', self.depth + 1))
        expanded.append(Board(Board.right(self.state), self, self.cost + 1, 'Right', self.depth + 1))

        expanded = [self for self in expanded if self.state != 0]

        return tuple(expanded)

    @staticmethod
    def goal():
        goal = []
        for i in range(Board.LENGTH * Board.LENGTH):
            goal.append(i)
        return tuple(goal)

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        eprint('usage: python driver.py [algorithm] [board]')
        sys.exit(1)
    board = Board(tuple([int(x) for x in sys.argv[2].split(',')]))
    algorithm = None
    if sys.argv[1] == 'bfs':
        algorithm = BreadthFirstSearch(board)
    elif sys.argv[1] == 'dfs':
        algorithm = DepthFirstSearch(board)
    elif sys.argv[1] == 'ast':
        algorithm = AStarSearch(board)
    else:
        print(sys.argv[1], sys.argv[1] == 'bfs')
        eprint('Invalid algorithm! Use: bsf, dfs or ast')

    final_board = algorithm.search()
    if final_board:
        output = open('output.txt','w')
        output.write('path_to_goal: %s\n' % final_board['moviments'])
        output.write('cost_of_path: %s\n' % len(final_board['moviments']))
        output.write('nodes_expanded: %s\n' % final_board['nodes_expanded'])
        output.write('search_depth: %s\n' % final_board['depth'])
        output.write('max_search_depth: %s\n' % Board.MAX_DEPTH)
        output.write('running_time: %s\n' % final_board['running_time'])
        output.write('max_ram_usage: %s\n' % final_board['max_ram_usage'])
        output.close()
#        print('path_to_goal: %s' % final_board['moviments'])
#        print('cost_of_path: %s' % len(final_board['moviments']))
#        print('nodes_expanded: %s' % final_board['nodes_expanded'])
#        print('search_depth: %s' % final_board['depth'])
#        print('max_search_depth: %s' % Board.MAX_DEPTH)
#        print('running_time: %s' % final_board['running_time'])
#        print('max_ram_usage: %s' % final_board['max_ram_usage'])
    else:
        eprint('Not found!')