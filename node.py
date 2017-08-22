import numpy as np
from functools import reduce
import bitarray
from bitarray import bitarray

class HTMNetwork:
    def __init__(self, loader, weights=None, shape=None, shape_max_chains=None):
        if shape is None:
            shape = [16*16, 16*4, 16]
        if weights is None:
            weights = [np.zeros(z) for z in zip(shape[:-1], shape[1:])]
        self.shape = shape
        self.weights = weights
        self.shape_max_chains = shape_max_chains
        self.loader = loader
        self.labels = []
        self.train = []
        self.movie = None

        self.network = [[Node() for _ in size] for size in self.shape]

    def generate_data(self, n):
        self.labels, self.train = self.loader.load_train()
        self.movie = self.loader.simple_movie(self.train)

    def start(self, n=1000):
        self.generate_data(n)
        for frame in self.movie:
            # numbers
            input = np.split(np.concatenate(np.array_split(frame, 16), 1), 16*16)
            for node in self.network[0]:
                node.



        #

class Node:
    def __init__(self, max_num_chains=5):
        self.max_num_chains = max_num_chains
        self.markov_graph = np.empty((0, 0))
        # mb 3d-matrix is better
        self.patterns = []
        # value
        self.proc_alpha = 0
        self.prev_pattern_index = None
        self.node_input = np.empty((0, 0))
        self.alpha = np.ones((1, 1))
        self.pattern_likelihood = np.empty(0)
        self.markov_chains = MarkovChains(max_num_chains)

# empty pattern is another pattern
    def input_to_pattern(self, node_input: np.array) -> np.array:

        self.node_input = node_input
        chosen_pos = np.argmax(node_input, 1)  # array of maximum's indexes

        pattern = np.zeros(node_input.shape, bool)
        for x in enumerate(chosen_pos):
            pattern[x] = (node_input[x] != 0)
        # its possible use array-indexes instead loop
        # pattern[range(len(pattern)), pattern] = (node_input[pattern] != 0)

        return pattern

    def process_forward(self, node_input, learn_mode=True):
        if(learn_mode):
            self.add_pattern(self.input_to_pattern(node_input))


    def calc_pattern_likelihood(self, node_input):
        #unpythonic
        self.pattern_likelihood = np.array([reduce(lambda x, y: x*y,
                                             node_input[pattern]) for pattern in self.patterns])
    def calc_feedforward_likelihood(self):
        ff_likelihood = np.zeros(self.max_num_chains)

        alpha = np.multiply(self.proc_alpha, self.pattern_likelihood)
        # numpy?
        for (pattern, chain) in enumerate(self.markov_chains.chain_nums):
            ff_likelihood[chain] += alpha[pattern]

        self.proc_alpha = self.alpha.dot(self.norm_markov_graph)

    def add_pattern(self, pattern):
        # analyze exception?
        if pattern not in self.patterns:
            np.pad(self.markov_graph, (0, 1), (0, 1), 'constant')
            self.patterns.append(pattern)
            index = len(pattern)

        else:
            index = self.patterns.index(pattern)

        prev_index = self.prev_pattern_index
        if(prev_index != None):
            self.markov_graph[prev_index, index]

        self.prev_pattern_index = index

# online clustering


class MarkovNode:
    def __init__(self, index, parent_index):
        self.index = index
        self.strongest_connect = 0
        self.parent = parent_index
        self.children = []

class MarkovChains:
    def __init__(self, max_num):
        self.num = 0
        self.max_num = max_num
        self.nodes = []
        self.chain_nums = []
        self.empty_chain = None

    def add_node(self, prev_index, cur_index):
        if self.num < self.max_num:
            self.nodes.append(MarkovNode(prev_index, cur_index))
            self.chain_nums.append(self.chain_nums[self.num if self.empty_chain is None else self.empty_chain])
        else:
            self.nodes.append(MarkovNode(prev_index, cur_index))
            self.chain_nums.append(self.chain_nums[prev_index])

    def strengthen_connect(self, prev_index, cur_index):
        # ignore new connections between until enough chains :(
        if (self.num < self.max_num):
            return

        new_connect = self.markov_graph[prev_index, cur_index]
        cur_node = self.nodes[cur_index]
        prev_node = self.nodes[prev_index]

        # probably unpythonic as fuck
        # need to be restructured

        if new_connect > prev_node.strongest_connect:
            self.reconnect(prev_node, cur_index)
            if self.chain_nums[prev_index] != self.chain_nums[cur_index]:
                # older chain is more likely to absorb younger one
                if prev_node.strongest_connect < cur_node.strongest_connect:
                    self.move(prev_node, self.chain_nums[cur_index])
                else:
                    self.move(cur_node, self.chain_nums[prev_index])

        if new_connect > cur_node.strongest_connect:
            self.reconnect(cur_node, prev_index)

    def reconnect(self, node, new_parent):
        # new connect can be only 1 stronger, I suppose
        node.strongest_connect += 1
        # do nothing in case of same parent
        if node.parent != new_parent:
            node.parent.children.remove(node.index)
            node.parent = new_parent
            node.parent.children.add(node.index)

    # move node and its children to another chain
    def move(self, node, dest_chain):
        self.chain_nums[node.index] = dest_chain
        for index in node.children:
            self.chain_nums[index] = dest_chain
