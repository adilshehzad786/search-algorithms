from __future__ import print_function
import sys

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        eprint("usage: python driver.py [method] [board]")
        sys.exit(1)

    board = []
    row = []
    actualIndex = 0
    for item in sys.argv[2].split(','):
        if actualIndex % 3 == 0:
            row = []
            board.append(row)
        row.append(item)
        actualIndex += 1
    print(board, sys.argv[2].split(','))