from collections import deque

import numpy as np


def calculate_steps_and_turns_to_goal(grid):
    n = len(grid)
    m = len(grid[0])

    # Initialize distance and turn grids with -1
    distance = [[-1 for _ in range(m)] for _ in range(n)]
    turns = [[-1 for _ in range(m)] for _ in range(n)]

    # Locate the position of G
    goal = None
    for i in range(n):
        for j in range(m):
            if grid[i][j] == 'G':
                goal = (i, j)
                break
        if goal:
            break

    if not goal:
        raise ValueError("Goal (G) not found in the grid.")

    # Directions: (dx, dy) and their corresponding labels
    directions = [(-1, 0, 'up'), (1, 0, 'down'), (0, -1, 'left'), (0, 1, 'right')]

    # BFS queue: (x, y, steps, turns, direction)
    queue = deque([(goal[0], goal[1], 0, 0, None)])
    distance[goal[0]][goal[1]] = 0
    turns[goal[0]][goal[1]] = 0

    while queue:
        x, y, steps, curr_turns, curr_dir = queue.popleft()

        for dx, dy, new_dir in directions:
            nx, ny = x + dx, y + dy

            # Check bounds and if the cell is traversable
            if 0 <= nx < n and 0 <= ny < m and grid[nx][ny] != 'H':
                # Calculate new turn count
                new_turns = curr_turns
                if curr_dir is not None and curr_dir != new_dir:
                    new_turns += 1

                # Only update if this path is shorter (steps) or fewer turns
                if distance[nx][ny] == -1 or steps + 1 < distance[nx][ny] or (
                        steps + 1 == distance[nx][ny] and new_turns < turns[nx][ny]):
                    distance[nx][ny] = steps + 1
                    turns[nx][ny] = new_turns
                    queue.append((nx, ny, steps + 1, new_turns, new_dir))

    distance = np.array(distance)
    turns = np.array(turns)
    return distance + turns

#
# # Example usage
# grid = [
#     ["F", "F", "H", "F"],
#     ["F", "F", "H", "F"],
#     ["H", "F", "F", "F"],
#     ["H", "H", "F", "G"],
# ]
# grid = [
#         "SFH",
#         "FHF",
#         "FFG",
#     ]
#
#
# distance_result, turns_result = calculate_steps_and_turns_to_goal(grid)
# total_result = np.array(distance_result) + np.array(turns_result)
#
#
#
# # Pretty print the results
# print("Steps to Goal:")
# for row in distance_result:
#     print(row)
#
# print("\nTurns to Goal:")
# for row in turns_result:
#     print(row)
#
# print("\nTotal Cost to Goal:")
# for row in total_result:
#     print(row)
