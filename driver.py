from __future__ import print_function
import sys
import copy
import time
import resource
import heapq

class Stack:
    """Stack implementation"""
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def __len__(self):
        return len(self.items)

class Queue:
    """Queue implementation"""
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0, item)

    def dequeue(self):
        return self.items.pop()

    def __len__(self):
        return len(self.items)

class PriorityQueue:
    """PriorityQueue implementation. Use priority and index to ensure that the first inputed will be returned first in case of same priority."""
    def __init__(self):
        self._queue = []
        self._index = 0

    def push(self, item, priority):
        heapq.heappush(self._queue, (priority, self._index, item))
        self._index += 1

    def pop(self):
        return heapq.heappop(self._queue)[-1]

    def __len__(self):
        return len(self._queue)

def memory_usage_resource():
    rusage_denom = 1024. * 1024.
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / rusage_denom
    return mem

class SearchAlgorithm:
    """Abstract implementation of a search algorithm"""
    def __init__(self, board):
        self.board = board
        self.init_frontier()
        self.nodes_expanded = 0
        self.max_search_depth = 0
    
    def order_expanded(self, list):
        """Change expanded order. The order can variate by the algorithm implementation"""
        return list

    def search(self):
        """Do the search"""
        start_time = time.time()
        visited = set()
        frontier_set = set()
        frontier_set.add(self.board)
        goal = Board.goal()
        nodes_expanded = 0
        while not self.is_frontier_empty():
            curr_node = self.get_next()
            visited.add(curr_node.state)

            if curr_node.state == goal:
                p_node = curr_node
                path = []
                while p_node.parent:
                    path.insert(0, p_node.moviment)
                    p_node = p_node.parent
                return {
                    'moviments': path,
                    'nodes_expanded': nodes_expanded,
                    'depth': curr_node.depth,
                    'running_time': time.time() - start_time,
                    'max_ram_usage': memory_usage_resource()
                }

            nodes_expanded += 1
            for expanded in self.order_expanded(curr_node.get_expanded()):
                if expanded.state not in visited and expanded.state not in frontier_set:
                    self.insert(expanded)
                    frontier_set.add(expanded.state)

class BreadthFirstSearch(SearchAlgorithm):
    """Breadth First Search implementation using Queue"""
    def init_frontier(self):
        self.frontier = Queue()
        self.frontier.enqueue(self.board)

    def insert(self, node):
        self.frontier.enqueue(node)

    def get_next(self):
        return self.frontier.dequeue()

    def is_frontier_empty(self):
        return len(self.frontier) == 0

class DepthFirstSearch(SearchAlgorithm):
    """Depth First Search implementation using Stack"""
    def init_frontier(self):
        self.frontier = Stack()
        self.frontier.push(self.board)

    def insert(self, node):
        self.frontier.push(node)

    def get_next(self):
        return self.frontier.pop()

    def is_frontier_empty(self):
        return len(self.frontier) == 0

    def order_expanded(self, l):
        """Reverse the order to ensure visiting in UDLR"""
        _l = list(l)
        _l.reverse()
        return _l

class AStarSearch(SearchAlgorithm):
    """A* Search implementation using Priority Queue"""
    def init_frontier(self):
        self.frontier = PriorityQueue()
        self.frontier.push(self.board, self.board.cost)
        self.goal = Board.goal()

    def insert(self, node):
        self.frontier.push(node, node.cost + node.distance(self.goal))

    def get_next(self):
        return self.frontier.pop()

    def is_frontier_empty(self):
        return len(self.frontier) == 0

class Board:
    """Board implementation"""
    LENGTH = 3
    MAX_DEPTH = 0
    def __init__(self, state, parent = None, cost = 0, moviment = None, depth = 0):
        self.state = state
        self.parent = parent
        self.cost = cost
        self.moviment = moviment
        self.depth = depth
        Board.MAX_DEPTH = max(Board.MAX_DEPTH, depth)

    @staticmethod
    def can_up(state):
        return state.index(0) >= Board.LENGTH

    LAST_ROW_INDEX = LENGTH * (LENGTH - 1)
    @staticmethod
    def can_down(state):
        return state.index(0) < Board.LAST_ROW_INDEX

    @staticmethod
    def can_left(state):
        return state.index(0) % Board.LENGTH != 0

    @staticmethod
    def can_right(state):
        return (state.index(0) + 1) % Board.LENGTH != 0

    @staticmethod
    def up(state):
        new_state = list(state)
        index = new_state.index(0)
        new_state[index - Board.LENGTH], new_state[index] = new_state[index], new_state[index - Board.LENGTH]
        return tuple(new_state)

    @staticmethod
    def down(state):
        new_state = list(state)
        index = new_state.index(0)
        new_state[index + Board.LENGTH], new_state[index] = new_state[index], new_state[index + Board.LENGTH]
        return tuple(new_state)

    @staticmethod
    def left(state):
        new_state = list(state)
        index = new_state.index(0)
        new_state[index - 1], new_state[index] = new_state[index], new_state[index - 1]
        return tuple(new_state)

    @staticmethod
    def right(state):
        new_state = list(state)
        index = new_state.index(0)
        new_state[index + 1], new_state[index] = new_state[index], new_state[index + 1]
        return tuple(new_state)

    @staticmethod
    def goal():
        """Create a static goal state"""
        goal = []
        for i in range(Board.LENGTH * Board.LENGTH):
            goal.append(i)
        return tuple(goal)

    @staticmethod
    def coordinates(value):
        return value % Board.LENGTH, value / Board.LENGTH

    def __str__(self):
        return self.state
    
    def get_expanded(self):
        """Retrieve all expanded nodes"""
        expanded = []
        next_cost = self.cost + 1
        next_depth = self.depth + 1
        if Board.can_up(self.state):
            expanded.append(Board(Board.up(self.state), self, next_cost, 'Up', next_depth))
        if Board.can_down(self.state):
            expanded.append(Board(Board.down(self.state), self, next_cost, 'Down', next_depth))
        if Board.can_left(self.state):
            expanded.append(Board(Board.left(self.state), self, next_cost, 'Left', next_depth))
        if Board.can_right(self.state):
            expanded.append(Board(Board.right(self.state), self, next_cost, 'Right', next_depth))

        return tuple(expanded)

    MAX_VALUE = (LENGTH * LENGTH) - 1
    def distance(self, state):
        """Calculate Manhattan distance"""
        sum = 0
        for value in range(1, Board.MAX_VALUE):
            x1, y1 = Board.coordinates(self.state.index(value))
            x2, y2 = Board.coordinates(state.index(value))
            sum += abs(x1 - x2) + abs(y1 - y2)
        return sum

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
    else:
        eprint('Not found!')