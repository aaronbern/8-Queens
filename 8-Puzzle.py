# Aaron Bernard
# CS441
# Programming Assignment #1


import heapq
import numpy as np

class Puzzle:
    def __init__(self, board, goal):
        self.board = np.array(board).reshape((3, 3))
        self.goal = np.array(goal).reshape((3, 3))

    def get_blank_pos(self):
        return np.argwhere(self.board == 0)[0]

    def is_goal(self):
        return np.array_equal(self.board, self.goal)

    def possible_moves(self):
        moves = []
        x, y = self.get_blank_pos()
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 3 and 0 <= ny < 3:
                new_board = self.board.copy()
                new_board[x, y], new_board[nx, ny] = new_board[nx, ny], new_board[x, y]
                moves.append(Puzzle(new_board.flatten(), self.goal))
        return moves

    def __eq__(self, other):
        return np.array_equal(self.board, other.board)

    def __hash__(self):
        return hash(str(self.board))

    def __lt__(self, other):
        return False

    def __str__(self):
        return str(self.board)


def manhattan_distance(puzzle):
    distance = 0
    for x in range(3):
        for y in range(3):
            value = puzzle.board[x, y]
            if value != 0:
                target_x, target_y = divmod(value - 1, 3)
                distance += abs(target_x - x) + abs(target_y - y)
    return distance


def misplaced_tiles(puzzle):
    return np.sum(puzzle.board != puzzle.goal) - 1


def count_reversals(puzzle):
    reversals = 0
    for x in range(3):
        for y in range(3):
            if x < 2:
                if puzzle.board[x, y] != 0 and puzzle.board[x + 1, y] != 0:
                    current_pos = (x, y)
                    next_pos = (x + 1, y)
                    if puzzle.board[current_pos] > puzzle.board[next_pos]:
                        reversals += 1
            if y < 2:
                if puzzle.board[x, y] != 0 and puzzle.board[x, y + 1] != 0:
                    current_pos = (x, y)
                    next_pos = (x, y + 1)
                    if puzzle.board[current_pos] > puzzle.board[next_pos]:
                        reversals += 1
    return reversals


def custom_heuristic(puzzle):
    return manhattan_distance(puzzle) + count_reversals(puzzle)


def get_inv_count(arr):
    inv_count = 0
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[j] != 0 and arr[i] != 0 and arr[i] > arr[j]:
                inv_count += 1
    return inv_count


def is_solvable(puzzle):
    inv_count = get_inv_count([tile for tile in puzzle])
    return inv_count % 2 == 0


def best_first_search(start, heuristic, max_steps=100000):
    if not is_solvable(start.board.flatten().tolist()):
        return None, 0, False

    frontier = [(heuristic(start), start)]
    came_from = {start: None}
    steps = 0

    while frontier and steps < max_steps:
        _, current = heapq.heappop(frontier)
        steps += 1  # Increment steps for each iteration
        if current.is_goal():
            return reconstruct_path(came_from, current), steps, True

        for neighbor in current.possible_moves():
            if neighbor not in came_from:
                came_from[neighbor] = current
                heapq.heappush(frontier, (heuristic(neighbor), neighbor))

    return None, steps, False


def a_star_search(start, heuristic, max_steps=100000):
    if not is_solvable(start.board.flatten().tolist()):
        return None, 0, False

    frontier = [(heuristic(start), 0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}
    steps = 0

    while frontier and steps < max_steps:
        _, current_cost, current = heapq.heappop(frontier)
        steps += 1  # Increment steps for each iteration
        if current.is_goal():
            return reconstruct_path(came_from, current), steps, True

        for neighbor in current.possible_moves():
            new_cost = current_cost + 1
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor)
                heapq.heappush(frontier, (priority, new_cost, neighbor))
                came_from[neighbor] = current

    return None, steps, False


def reconstruct_path(came_from, current):
    path = []
    while current:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path


def display_path(path):
    return " → ".join([str(p.board.flatten()) for p in path])


if __name__ == "__main__":
    start_states = [
        [1, 2, 3, 4, 5, 6, 8, 7, 0],  # Unsolvable
        [1, 2, 3, 4, 5, 6, 0, 7, 8],  
        [4, 1, 3, 7, 2, 6, 5, 8, 0],  
        [1, 2, 0, 4, 5, 3, 7, 8, 6], 
        [1, 2, 3, 4, 0, 5, 7, 8, 6]   
    ]
    goal_state = [1, 2, 3, 4, 5, 6, 7, 8, 0]
    heuristics = [manhattan_distance, misplaced_tiles, custom_heuristic]
    heuristic_names = ["Manhattan Distance", "Misplaced Tiles", "Custom Heuristic"]
    search_algorithms = [best_first_search, a_star_search]
    search_algorithm_names = ["Best-first search", "A* search"]

    import sys
    import re

    if len(sys.argv) > 1:
        input_puzzle = sys.argv[1]
        if input_puzzle.lower() == "example":
            for i, heuristic in enumerate(heuristics):
                for j, search_algorithm in enumerate(search_algorithms):
                    print(f"\n{search_algorithm_names[j]} with {heuristic_names[i]}:")
                    avg_steps = 0
                    for start in start_states:
                        puzzle = Puzzle(start, goal_state)
                        path, steps, solvable = search_algorithm(puzzle, heuristic)
                        if solvable:
                            avg_steps += steps
                            print(display_path(path))
                        else:
                            print("Not solvable")
                    print(f"Average number of steps (search iterations): {avg_steps / len(start_states)}")
        else:
            match = re.match(r'\[([0-8,\s]+)\]', input_puzzle)
            if match:
                input_puzzle = list(map(int, match.group(1).split(',')))
                if len(input_puzzle) != 9 or set(input_puzzle) != set(range(9)):
                    print("Invalid puzzle input. Must be a list of 9 numbers from 0 to 8.")
                    sys.exit(1)
                print("\nAvailable heuristics:")
                for idx, name in enumerate(heuristic_names):
                    print(f"{idx}: {name}")

                print("\nAvailable search algorithms:")
                for idx, name in enumerate(search_algorithm_names):
                    print(f"{idx}: {name}")

                heuristic_idx = int(input("\nSelect a heuristic (0, 1, or 2): "))
                algorithm_idx = int(input("Select a search algorithm (0 or 1): "))

                heuristic = heuristics[heuristic_idx]
                search_algorithm = search_algorithms[algorithm_idx]

                puzzle = Puzzle(input_puzzle, goal_state)
                path, steps, solvable = search_algorithm(puzzle, heuristic)
                if solvable:
                    print(display_path(path))
                    print(f"Number of steps (search iterations): {steps}")
                else:
                    print("Not solvable")
            else:
                print("Invalid input format. Please provide a list of 9 numbers from 0 to 8.")
    else:
        print("Please provide a puzzle input or type 'example' to run the example puzzles.")
