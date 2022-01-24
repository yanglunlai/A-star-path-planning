import numpy as np
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker
from pybullet_tools.utils import connect, disconnect, get_joint_positions, wait_if_gui, set_joint_positions, joint_from_name, get_link_pose, link_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS
import time
from queue import PriorityQueue
########## A* algorithm for pr2 mobile robot by Yang-Lun Lai ###############

def main(screenshot=False):
    # initialize PyBullet
    connect(use_gui=False)
    # load robot and obstacle resources
    robots, obstacles = load_env('pr2doorway.json')

    # define active DoFs
    base_joints = [joint_from_name(robots['pr2'], name) for name in PR2_GROUPS['base']]

    collision_fn = get_collision_fn_PR2(robots['pr2'], base_joints, list(obstacles.values()))
    # Example use of collision checking
    # print("Robot colliding? ", collision_fn((0.5, -1.3, -np.pi/2))) # True/False

    # Example use of setting body poses
    # set_pose(obstacles['ikeatable6'], ((0, 0, 0), (1, 0, 0, 0)))

    # Example of draw 
    # draw_sphere_marker((0, 0, 1), 0.1, (1, 0, 0, 1))
    print("start searching the path using A*...")
    start_config = tuple(get_joint_positions(robots['pr2'], base_joints))
    goal_config = (2.6, -1.3, -np.pi/2)
    path = []
    start_time = time.time()
    ### YOUR CODE HERE ###
    class Node:
        def __init__(self, x_in, y_in, theta_in, id_in, parentid_in, distance_in):
            self.x = x_in
            self.y = y_in
            self.theta = theta_in
            self.id = id_in
            self.parentid = parentid_in
            self.distance = distance_in

    def neighbors(config, step_sz, num):
        x = config[0]
        y = config[1]
        successor = []
        if num == 4:
            successor = [(x + step_sz, y), (x - step_sz, y), (x, y + step_sz), (x, y - step_sz)]
        if num == 8:
            successor = [(x + step_sz, y), (x - step_sz, y), (x, y + step_sz), (x, y - step_sz),
                         (x + step_sz, y + step_sz), (x - step_sz, y - step_sz), (x - step_sz, y + step_sz),
                         (x + step_sz, y - step_sz)]
        return successor

    def c(n_config, m_config):
        nx = n_config[0]
        ny = n_config[1]
        nt = n_config[2]
        mx = m_config[0]
        my = m_config[1]
        mt = m_config[2]
        return np.sqrt((nx - mx) ** 2 + (ny - my) ** 2 + (min(abs(nt - mt), 2 * np.pi - abs(nt - mt))) ** 2)

    def h(n_config, g_config):
        nx = n_config[0]
        ny = n_config[1]
        nt = n_config[2]
        gx = g_config[0]
        gy = g_config[1]
        gt = g_config[2]
        return np.sqrt((nx - gx) ** 2 + (ny - gy) ** 2 + (min(abs(nt - gt), 2 * np.pi - abs(nt - gt))) ** 2)

    q = PriorityQueue()
    step_size = 0.1
    theta_size = 2*np.pi/4
    current_config = start_config
    closed = [] # The closed list is a collection of all expanded nodes.
    free = [] # the collision-free configurations
    collision = [] # the colliding configurations
    closed.append(current_config)
    que_ind = 0
    child_id = 0
    parent_initial = -1
    g = 0
    f = g + h(current_config, goal_config) 
    current_node = Node(current_config[0],current_config[1], current_config[2], child_id, parent_initial, g)
    q.put((f, que_ind, current_node))

    while not q.empty():
        current_neighbor = neighbors(current_config, step_size, 8)
        parent = Node(current_config[0],current_config[1], current_config[2], child_id, current_node, g)
        for neighbor_id in current_neighbor:
            for theta_ind in range(8):
                next = (neighbor_id[0], neighbor_id[1], theta_size * theta_ind)
                if next not in closed:
                    closed.append(next)
                    if collision_fn(next) == False:
                        free.append(next)
                        child_id = child_id + 1
                        que_ind = que_ind + 1
                        g = parent.distance + c(current_config,next)
                        f = g + h(next,goal_config)
                        new_node = Node(next[0],next[1], next[2], child_id, parent, g)
                        q.put((f, que_ind, new_node))
                    else: 
                        collision.append(next)

        current_node = q.get()[2]
        g = current_node.distance
        current_config = (current_node.x, current_node.y, current_node.theta)
        if h(current_config, goal_config) < step_size:
            print ("the path cost = {}".format(current_node.distance))
            while current_node.id != 0:
                current_config = (current_node.x, current_node.y, current_node.theta)
                path.insert(0, current_config)
                current_node = current_node.parentid
            break
    searching_end = time.time()
    print("Planner run time(without drawing the points): ", time.time() - start_time)
    disconnect()
    ## drawing the search space and the path
    connect(use_gui=True)
    # load robot and obstacle resources
    robots, obstacles = load_env('pr2doorway.json')
    # draw the free space in blue and draw the collision space in red
    drawn_free = []
    drawn_collision = []
    for free_i in free:
        if (free_i[0],free_i[1]) not in drawn_free:
            draw_sphere_marker((free_i[0],free_i[1],0), 0.1, (0, 0, 1, 1))
        drawn_free.append((free_i[0],free_i[1]))

    for collision_i in collision:
        if (collision_i[0], collision_i[1]) not in drawn_collision:
            draw_sphere_marker((collision_i[0],collision_i[1],0.03), 0.1, (1, 0, 0, 1))
        drawn_collision.append((collision_i[0],collision_i[1]))
    # draw the path in black
    drawn_path = []
    for path_i in path:
        if (path_i[0],path_i[1]) not in drawn_path:
            draw_sphere_marker((path_i[0],path_i[1],0.08), 0.1, (0, 0, 0, 1))
        drawn_path.append((path_i[0],path_i[1]))

    ######################
    print("Planner run time: ", time.time() - start_time)
    # Execute planned path
    execute_trajectory(robots['pr2'], base_joints, path, sleep=0.2)
    # Keep graphics window opened
    wait_if_gui()
    disconnect()

if __name__ == '__main__':
    main()
