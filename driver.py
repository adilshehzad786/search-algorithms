from __future__ import print_function
import sys
import copy
import time
import resource
from collections import deque

def wait():
    raw_input('')

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

    def search(self):
        start_time = time.time()
        checked_boards = set()
        while not self.is_frontier_empty():
            state = self.get_next()
            print(state, state.moviments)
            checked_boards.add(state.signature())

            self.max_ram_usage = max(self.max_ram_usage, memory_usage_resource())
            if state.is_goal_reached():
                self.running_time = time.time() - start_time
                return state

            self.nodes_expanded += 1
            for moviment in state.possible_moviments():
                new_state = state.clone().apply_moviment(moviment)
                self.max_search_depth = max(self.max_search_depth, len(new_state.moviments))
                if new_state.signature() not in checked_boards:
                    print(new_state.signature(), checked_boards, new_state.signature() not in checked_boards)
                    wait()
                    self.insert(new_state)

        self.running_time = time.time() - start_time
        return None

class BreadthFirstSearch(SearchAlgorithm):
    def init_frontier(self):
        self.frontier = deque([self.board])

    def insert(self, state):
        self.frontier.append(state)

    def get_next(self):
        return self.frontier.popleft()

    def is_frontier_empty(self):
        return len(self.frontier) == 0

class DepthFirstSearch(SearchAlgorithm):
    def init_frontier(self):
        self.frontier = [self.board]

    def insert(self, state):
        self.frontier.append(state)

    def get_next(self):
        return self.frontier.pop()

    def is_frontier_empty(self):
        return len(self.frontier) == 0

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
    def __init__(self, description):
        self.rows = []
        actualIndex = 0
        row = []
        for item in description.split(','):
            if actualIndex % 3 == 0:
                row = []
                self.rows.append(row)
            if item == '0':
                self.x = actualIndex % 3
                self.y = len(self.rows) - 1
            row.append(int(item))
            actualIndex += 1
        self.moviments = []
    def __str__(self):
        return self.signature()

    def possible_moviments(self):
        moviments = []
        if self.y > 0:
            moviments.append('Up')
        if self.y < 2:
            moviments.append('Down')
        if self.x > 0:
            moviments.append('Left')
        if self.x < 2:
            moviments.append('Right')
        return moviments

    def apply_moviment(self, moviment):
        self.moviments.append(moviment)
        if moviment == 'Up':
            aux = self.rows[self.y][self.x]
            self.rows[self.y][self.x] = self.rows[self.y - 1][self.x]
            self.rows[self.y - 1][self.x] = aux
            self.y -= 1
        elif moviment == 'Down':
            aux = self.rows[self.y][self.x]
            self.rows[self.y][self.x] = self.rows[self.y + 1][self.x]
            self.rows[self.y + 1][self.x] = aux
            self.y += 1
        elif moviment == 'Right':
            aux = self.rows[self.y][self.x]
            self.rows[self.y][self.x] = self.rows[self.y][self.x + 1]
            self.rows[self.y][self.x + 1] = aux
            self.x += 1
        elif moviment == 'Left':
            aux = self.rows[self.y][self.x]
            self.rows[self.y][self.x] = self.rows[self.y][self.x - 1]
            self.rows[self.y][self.x - 1] = aux
            self.x -= 1
        return self

    def is_goal_reached(self):
        curr_value = 0
        for row in self.rows:
            for item in row:
                if item != curr_value:
                    return False
                curr_value += 1
        return True

    def signature(self):
        return ','.join(str(item) for innerlist in self.rows for item in innerlist)

    def clone(self):
        return copy.deepcopy(self)


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        eprint('usage: python driver.py [algorithm] [board]')
        sys.exit(1)
    board = Board(sys.argv[2])
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
        print('path_to_goal: %s' % final_board.moviments)
        print('cost_of_path: %s' % len(final_board.moviments))
        print('nodes_expanded: %s' % algorithm.nodes_expanded)
        print('search_depth: %s' % len(final_board.moviments))
        print('max_search_depth: %s' % algorithm.max_search_depth)
        print('running_time: %s' % algorithm.running_time)
        print('max_ram_usage: %s' % algorithm.max_ram_usage)
    else:
        print('Not found!')