from enum import Enum
from copy import deepcopy
import time
import heapq

class NodeType(Enum):
    student = 0
    pizza = 1
    normal = 2

class Edge:
    id: int
    nodes = set
    is_loose: bool
    wait_time: int

    def __init__(self, id, v1, v2):
        self.id = id
        self.nodes = (v1, v2)
        self.is_loose = False
        self.wait_time = 0

class Node:
    id: int
    edges: dict # access to edge via id of the other connected node
    type: NodeType

    def __init__(self, id):
        self.id = id
        self.edges = dict()
        self.type = NodeType.normal

    def add_edge(self, u, edge: Edge):
        self.edges[u] = edge

class Graph:
    num_of_nodes: int 
    num_of_edges: int
    nodes: list[Node] # with n elements
    edges: list[Edge]
    student_pizza: dict # access to id of the pizza node which a node wants (by their id)
    pizza_student: dict # access to id of the student node
    loose_edges: set[Edge]
    ziraj_first_position: int
    priorities: list # access to prority of each node
    students: set
    pizzas: set

    def __init__(self, n, m):
        self.num_of_nodes = n
        self.num_of_edges = m
        self.nodes = [Node(i) for i in range(self.num_of_nodes)] # id of nodes is from 0 up to n-1
        self.edges = []
        self.student_pizza = dict()
        self.pizza_student = dict()
        self.loose_edges = set()
        self.priorities = [[] for i in range(self.num_of_nodes)]
        self.students = set()
        self.pizzas = set()
        
    def add_edge(self, id, u, v):
        edge = Edge(id, v, u)
        self.edges.append(edge)

        self.nodes[u].add_edge(v, edge)
        self.nodes[v].add_edge(u, edge)

    def change_edge_to_loose(self, id, xi):
        self.edges[id].is_loose = True
        self.edges[id].wait_time = xi

        self.loose_edges.add(self.edges[id])

    def assign_pizza_and_student(self, student, pizza):
        self.nodes[student].type = NodeType.student
        self.students.add(student)
        self.nodes[pizza].type = NodeType.pizza
        self.pizzas.add(pizza)

        self.student_pizza[student] = pizza
        self.pizza_student[pizza] = student

    def add_prority(self, u, v):
        self.priorities[v].append(u)

class State:
    ziraj_current_position: int
    loose_edges_last_pass_time: dict # access to last pass time of each loose edge by its id
    students_with_pizza: set # set of id of studesnt who delivered their pizza
    delivered_pizza: set # set of id pizzas wiche are delivered
    path: list # list of id of nodes on the crrent path
    passed_time: int
    holded_pizza: int
    stay_on_node: int

    def __init__(self, ziraj_position):
        self.ziraj_current_position = ziraj_position
        self.loose_edges_last_pass_time = dict()
        self.students_with_pizza = set()
        self.delivered_pizza = set()
        self.path = [ziraj_position]
        self.passed_time = 0
        self.holded_pizza = None
        self.stay_on_node = 0

    def update_pass_time_of_loose_edges(self):
        for loose_edge in list(self.loose_edges_last_pass_time.keys()):
            self.loose_edges_last_pass_time[loose_edge] -= 1
            if self.loose_edges_last_pass_time[loose_edge] == 0:
                del self.loose_edges_last_pass_time[loose_edge]

    def __eq__(self, s: object):
        return (
            self.ziraj_current_position == s.ziraj_current_position and
            self.stay_on_node == s.stay_on_node and
            self.delivered_pizza == s.delivered_pizza and
            self.holded_pizza == s.holded_pizza
        )
    
    def __str__(self) -> str:
        return (
            str(self.ziraj_current_position) +
            str(self.stay_on_node) +
            str(self.delivered_pizza) +
            str(self.holded_pizza)
        )

    def __hash__(self) -> int:
        return hash(self.__str__())

    def show(self):
        print("-------------------------------")
        print(self.ziraj_current_position, self.passed_time, self.holded_pizza)
        for edge in self.loose_edges_last_pass_time.keys():
            print(edge, ":", self.loose_edges_last_pass_time[edge], end = ', ')
        print("")
        print(len(self.students_with_pizza))
        print(self.path)

def get_all_next_states(current_state: State):
    out = []

    ziraj : Node = graph.nodes[current_state.ziraj_current_position]

    for neighbor in graph.nodes[current_state.ziraj_current_position].edges.keys():
        edge: Edge = graph.nodes[current_state.ziraj_current_position].edges[neighbor]

        next_state = deepcopy(current_state)
        next_state.update_pass_time_of_loose_edges()
        next_state.passed_time += 1

        # You cannot pass this edge
        if edge.is_loose:
            if edge.id in current_state.loose_edges_last_pass_time.keys():
                next_state.path.append(ziraj.id)
                next_state.stay_on_node += 1
                out.append(next_state)
                continue
            else: # You can
                next_state.loose_edges_last_pass_time[edge.id] = graph.edges[edge.id].wait_time
        
        next_state.ziraj_current_position = neighbor
        next_state.path.append(neighbor)
        next_state.stay_on_node = 0

        if next_state.holded_pizza != None: # ziraj holds a pizza
            if graph.nodes[neighbor].type == NodeType.student and graph.student_pizza[neighbor] == next_state.holded_pizza:
                next_state.students_with_pizza.add(neighbor)
                next_state.delivered_pizza.add(next_state.holded_pizza)
                next_state.holded_pizza = None
        if next_state.holded_pizza == None:
            if graph.nodes[neighbor].type == NodeType.pizza and not neighbor in next_state.delivered_pizza:
                for student in graph.priorities[graph.pizza_student[neighbor]]:
                    if not student in next_state.students_with_pizza:
                        break
                else:
                    next_state.holded_pizza = neighbor

        out.append(next_state)

    return out

def bfs():
    frontier = [init_state] # as a queue
    explored = set()
    visited_states = 1

    while len(frontier) != 0:
        current_state = frontier.pop(0)

        for next_state in get_all_next_states(current_state):
            if (next_state in frontier) or (next_state in explored):
                continue

            visited_states += 1
            
            if len(next_state.delivered_pizza) == num_of_orders:
                return next_state.path, visited_states
            
            frontier.append(next_state)

        explored.add(current_state)
    
    return None, None

def depth_limited_search(l):
    frontier: list[State] = [init_state] # as a stack
    explored_at: dict[State, int] = dict()
    visited_states = 1

    while len(frontier) != 0:
        current_state = frontier.pop()
        depth = current_state.passed_time
        
        for next_state in get_all_next_states(current_state):
            if (next_state in frontier) or (next_state in explored_at.keys() and explored_at[next_state] <= depth+1):
                continue
            
            if next_state.passed_time >= l:
                continue

            visited_states += 1

            if len(next_state.delivered_pizza) == num_of_orders:
                return next_state.path, visited_states
            
            frontier.append(next_state)
        
        explored_at[current_state] = depth
        
    return None, visited_states

def ids():
    l = 0
    visited_states = 0

    while True:
        path, more_visited_states = depth_limited_search(l)
        visited_states += more_visited_states

        if path != None:
            return path, visited_states
        else:
            l += 1

class HeapNode:
    state: State
    heuristic: int
    f: int

    def __init__(self, s: State, alpha):
        self.state = s
        
        self.heuristic = (2 * num_of_orders) - (2 * len(self.state.delivered_pizza))

        if s.holded_pizza != None:
            self.heuristic -= 1

        self.f = alpha*self.heuristic + self.state.passed_time

    def __lt__(self, s):
        return self.f < s.f

    def __eq__(self, s: State):
        return self.state == s

def a_star(alpha = 1):
    frontier = [] # as a min heap
    explored = set()

    heapq.heappush(frontier, HeapNode(init_state, alpha))
    visited_states = 1

    while len(frontier) != 0:
        current_state = heapq.heappop(frontier).state
        
        for next_state in get_all_next_states(current_state):
            if next_state in explored or next_state in frontier:
                continue

            visited_states += 1

            if len(next_state.delivered_pizza) == num_of_orders:
                return next_state.path, visited_states
            
            heapq.heappush(frontier, HeapNode(next_state, alpha))
            
        explored.add(current_state)

    return None, None

def read_input(i):
    f = open("Tests/Test"+str(i)+".txt", 'r', encoding='utf-8')

    n, m = [int(x) for x in f.readline().split()]

    graph = Graph(n, m)

    for i in range(m):
        u, v = [int(x) for x in f.readline().split()]
        graph.add_edge(i, v-1, u-1)

    num_of_loose_edges = int(f.readline())

    for i in range(num_of_loose_edges):
        id, xi = [int(x) for x in f.readline().split()]
        graph.change_edge_to_loose(id-1, xi)

    graph.ziraj_first_position = int(f.readline())-1
    num_of_students = int(f.readline())

    students = []

    for i in range(num_of_students):
        student, pizza = [int(x) for x in f.readline().split()]
        graph.assign_pizza_and_student(student-1, pizza-1)
        students.append(student-1)

    num_of_priority_rules = int(f.readline())

    for i in range(num_of_priority_rules):
        a, b = [int(x) for x in f.readline().split()]
        graph.add_prority(students[a-1], students[b-1])

    f.close()
    return graph

def print_answer(path, run_time, visited_states):
    print("---------------------------------------------")
    if path == None:
        print("There is no way!")
        return
    
    print("Best time:               ", len(path)-1)
    print("Suggested Path:          ", end=' ')

    for i in range(len(path)):
        if(i != len(path)-1):
            print(path[i]+1, end=' > ')
        else:
            print(path[i]+1, end='')

    print("")
    print("Run Time:                ", run_time, "s")
    print("Number of Visited States:", visited_states)

    print("---------------------------------------------")

selected_algorithm = -1
while selected_algorithm not in [1,2,3]:
    selected_algorithm = int(input("Select the search algorithm you want:\n [1] BFS\n [2] IDS\n [3] A*\n"))

a_star_alpha = 1
if selected_algorithm == 3:
    a_star_alpha = float(input("Enter amount of alpha: "))

graph = read_input(int(input("Enter number of test file: ")))
num_of_orders = len(graph.student_pizza.keys())

init_state = State(graph.ziraj_first_position)  

start_time = time.time()

if selected_algorithm == 1:
    answer, visited_states = bfs()
elif selected_algorithm == 2:
    answer, visited_states = ids()
elif selected_algorithm == 3:
    answer, visited_states = a_star(a_star_alpha)

end_time = time.time()

print_answer(answer, end_time - start_time, visited_states)