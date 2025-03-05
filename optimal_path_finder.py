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


def random_map_generator(size, holes):
    from games.custom_grid import SimpleEnv

    custom_map = SimpleEnv.get_random_map(size, holes)

    step_grid = calculate_steps_and_turns_to_goal(custom_map)
    # list all positions which you can reach from the goal
    reachables = [(y, x) for y in range(len(step_grid)) for x in range(len(step_grid[y])) if step_grid[y][x] > 0]

    # if not more than 20% of all positions are reachable, reroll (goal locked between holes)
    if not len(reachables) / size ** 2 > 0.2:
        return random_map_generator(size, holes)

    start_pos = reachables[np.random.randint(0, len(reachables))]

    return custom_map, start_pos


#
# if __name__ == "__main__":
#     num_maps = 1000
#     maps = []
#
#     # Create the custom game
#     for i in range(num_maps):
#         random_map, start_pos = random_map_generator(4, 3)
#         direction = np.random.randint(0, 4)
#         game_dict = {
#             "map": random_map.tolist(),
#             "start_pos": start_pos,
#             "start_dir": direction
#         }
#         # turn to string
#         maps.append(game_dict)
#
#     # save to json file
#     import json
#     with open("custom_maps/4x4.json", "w") as f:
#         json.dump(maps, f)
#
#     print("Maps saved to custom_maps.json")
#
#     # open maps and try to retrive one
#     with open("custom_maps/4x4.json", "r") as f:
#         maps = json.load(f)
#
#     print("Maps loaded from custom_maps.json")
#     print(maps[3])


