# -*- coding: utf-8 -*-
"""
Based on the implementation of the following paper:
"RiWalk: Fast Structural Node Embedding via Role Identification", ICDM, 2019
"""
from collections import deque
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import math


class RiGraph:
    def __init__(self, nx_g, n2v_g, node_attr, args):
        self.g = nx_g
        self.n2v_g = n2v_g
        self.until = args.until_k
        self.num_walks = args.num_walks
        self.walk_length = args.walk_length
        self.workers = args.workers
        self.flag = args.flag
        # self.discount = not args.without_discount
        self.discount = True  # always use discount version for now!
        logging.debug('discount option: {}'.format('On' if self.discount else 'Off'))

        self.num_nodes = len(self.g)
        # degree
        self.degrees_ = tuple([len(self.g[_]) for _ in range(len(self.g))])
        # self.logDegrees_ = tuple(np.asarray(self.degrees_).tolist())
        self.logDegrees_ = tuple(np.log2(np.asarray(self.degrees_) + 1).astype(np.int32).tolist())

        # flag: 'na'
        if args.flag == 'na':
            self.node_attr_ = tuple(np.asarray(node_attr).astype(np.int32).tolist())
            self.log_node_attr_ = tuple(np.log2(np.asarray(self.node_attr_) + 1).astype(np.int32).tolist())

    def process_the_last_layer(self, root, the_last_layer, visited, tmp_list):
        """
        process each {node} in {the_last_layer} and make the first {tmp_list[node]} neighbors of {node} are
        within {self.until} hops from {root}
        such that when simulating random walks we can randomly choose nodes in {self.g[node][:tmp_list[node]]}
        to make sure walking within {self.until}-hops from {root}.
        :param root: the anchor node
        :param the_last_layer: a list of nodes which are exactly {self.until} hops from {root}
        :param visited: a list of length {self.num_nodes}.
                        visited[node]==root indicating node is within {self.until} hops from root
        :param tmp_list: a list of length {self.num_nodes}.
        :return:
        """
        for node in the_last_layer:
            neighbors = self.g[node]
            ln = len(neighbors) - 1
            p, q = 0, ln
            while p <= q:
                while q >= 0 and visited[neighbors[q]] != root:
                    q -= 1
                while p <= ln and visited[neighbors[p]] == root:
                    p += 1
                if p < q:
                    neighbors[p], neighbors[q] = neighbors[q], neighbors[p]
            tmp_list[node] = p

    def bfs(self, root, tmp_list, visited):
        """
        breadth-first search {self.g} from {root} for {self.until} hops.
        mark each {node} within {self.until} hops from {root} with {visited[node]=root}.
        return dictionary of {node: its_distance_from_root} pairs.
        :param root: the anchor node.
        :param tmp_list: make sure the first {tmp_list[node]} neighbors of {node}
                         are within {self.until} hops from {root}.
        :param visited: make sure nodes within {self.until} hops from {root}
                        being marked with {visited[node]=root}.
        :return: distance_dict
        """
        g = self.g
        last_layer = deque()
        last_layer.append(root)
        distance_dict = {root: 0}
        visited[root] = root
        layer_count = 1
        while True:
            next_layer = deque()
            for node_ in last_layer:
                neighbors = g[node_]
                for nb in neighbors:
                    if visited[nb] != root:
                        visited[nb] = root
                        next_layer.append(nb)
                tmp_list[node_] = len(neighbors)
            if len(next_layer) == 0:
                return distance_dict
            for _ in next_layer:
                distance_dict[_] = layer_count
            if layer_count == self.until:
                self.process_the_last_layer(root, next_layer, visited, tmp_list)
                return distance_dict
            last_layer = next_layer
            layer_count += 1


    def bfs_na(self, root, tmp_list, visited, n_attr):
        """
        * for the 'na' version
        breadth-first search {self.g} from {root} for {self.until} hops.
        mark each {node} within {self.until} hops from {root} with {visited[node]=root}.
        return dictionary of {node: its_distance_from_root} pairs.
        :param root: the anchor node.
        :param tmp_list: make sure the first {tmp_list[node]} neighbors of {node}
                         are within {self.until} hops from {root}.
        :param visited: make sure nodes within {self.until} hops from {root}
                        being marked with {visited[node]=root}.
        :return: distance_dict
        """
        g = self.g
        last_layer = deque()
        last_layer.append(root)
        distance_dict = {root: 0}
        n_attribute_dict = {root: n_attr[root]}
        visited[root] = root
        layer_count = 1
        while True:
            next_layer = deque()
            for node_ in last_layer:
                neighbors = g[node_]
                for nb in neighbors:
                    if visited[nb] != root:
                        visited[nb] = root
                        next_layer.append(nb)
                tmp_list[node_] = len(neighbors)
            if len(next_layer) == 0:
                return distance_dict, n_attribute_dict
            for _ in next_layer:
                distance_dict[_] = layer_count
                n_attribute_dict[_] = n_attr[_]
            if layer_count == self.until:
                self.process_the_last_layer(root, next_layer, visited, tmp_list)
                return distance_dict, n_attribute_dict
            last_layer = next_layer
            layer_count += 1

    def get_sp_dict(self, root, node_layer_dict, nb_nodes):
        """
        given a list of neighbor nodes {nb_nodes},
        return a dictionary of their new identifiers calculated by RiWalk-SP.
        :param root: the anchor node.
        :param node_layer_dict: a dictionary of {node: its_distance_from_root} pairs.
        :param nb_nodes: a list of neighbor nodes of root (with order).
        :return: sp_dict, a dictionary of {node: its_new_identifier} pairs
        """
        layer_list = [node_layer_dict[node] for node in nb_nodes]
        if self.discount:
            root_degree = self.logDegrees_[root]
            degree_list = [self.logDegrees_[node] for node in nb_nodes]
        else:
            root_degree = self.degrees_[root]
            degree_list = [self.degrees_[node] for node in nb_nodes]
        sp_dict = {node_: hash((root_degree, layer_, degree_)) for node_, layer_, degree_ in
                   zip(nb_nodes, layer_list, degree_list)}
        # # my version of concatenation!
        # sp_dict = {node_: int(str(root_degree)+str(layer_)+str(degree_)) for node_, layer_, degree_ in
        #            zip(nb_nodes, layer_list, degree_list)}
        return sp_dict

    def get_wl_dict(self, root, node_layer_dict, nb_nodes, wl_lists, tmp_list):
        """
        given a list of neighbor nodes {nb_nodes},
        return a dictionary of their new identifiers calculated by RiWalk-WL.
        :param root: the anchor node.
        :param node_layer_dict: a dictionary of {node: its_distance_from_root} pairs.
        :param nb_nodes: a list of neighbor nodes of root (with order).
        :param wl_lists: temp memory space of size {self.num_nodes * (self.until +1)}
        :return: wl_dict, a dictionary of {node: its_new_identifier} pairs
        """
        g = self.g
        layer_list = [node_layer_dict[node] for node in nb_nodes]
        for _ in nb_nodes:
            wl_lists[_] = [0] * (self.until + 1)
        for i, nb in enumerate(nb_nodes):
            layer = layer_list[i]
            cur_nbrs=g[nb]
            tl = tmp_list[nb]
            for l in range(tl):  # time consuming
                wl_lists[cur_nbrs[l]][layer] += 1
        x_lists = [wl_lists[_] for _ in nb_nodes]
        if self.discount:
            x_lists = np.log2(np.asarray(x_lists) + 1).astype(np.int32).tolist()
        root_x_ = tuple(x_lists[nb_nodes.index(root)])
        wl_dict = {node_: hash((root_x_, layer_, tuple(x_))) for node_, layer_, x_ in
                   zip(nb_nodes, layer_list, x_lists)}
        return wl_dict

    def get_na_dict(self, root, node_layer_dict, nb_nodes, node_attr_dict, wl_lists, tmp_list):
        """
        given a list of neighbor nodes {nb_nodes},
        return a dictionary of their new identifiers calculated by 'NSA: Node Single Attribute'.
        :param root: the anchor node.
        :param node_layer_dict: a dictionary of {node: its_distance_from_root} pairs.
        :param nb_nodes: a list of neighbor nodes of root (with order).
        :return: na_dict, a dictionary of {node: its_new_identifier} pairs
        """
        layer_list = [node_layer_dict[node] for node in nb_nodes]
        attr_list = [node_attr_dict[node] for node in nb_nodes]
        root_x_ = tuple(attr_list[nb_nodes.index(root)])
        na_dict = {node_: hash((root_x_, layer_, tuple(x_))) for node_, layer_, x_ in
                   zip(nb_nodes, layer_list, attr_list)}

        # extra
        wl_dict = self.get_wl_dict(root, node_layer_dict, nb_nodes, wl_lists, tmp_list)
        na_dict_extra = {node_: hash((na_dict[node_], wl_dict[node_])) for node_ in nb_nodes}

        return na_dict_extra

    def simulate_walk(self, walk_length, root, tmp_list):
        """
        simulating random walks from {root}.
        :param walk_length: the max length of a random walk.
        :param root: each random walk starts from {root}
        :param tmp_list: the first {tmp_list[node]} neighbors of {node} are
                         all within {self.until} hops from {root}.
                         suppose the last node of our walk is {cur},
                         we choose the next hop in {self.g[cur][:tmp_list[cur]]}
                         such that we will never walk out of {self.until} hops from root.
        :return: walk
        """
        walk = [root]
        # g = self.g
        n2v_g = self.n2v_g
        # rand = np.random.randint(self.num_nodes, size=walk_length)
        # for i in range(walk_length - 1):
        while len(walk) < walk_length:
            cur = walk[-1]
            # cur_nbrs = g[cur]
            tl = tmp_list[cur]
            if tl:
                # next_node = cur_nbrs[rand[i] % tl]
                # walk.append(next_node)

                walk_options = n2v_g.d_graph[walk[-1]].get(n2v_g.NEIGHBORS_KEY, None)
                # Skip dead end nodes
                if not walk_options:
                    break

                if len(walk) == 1:  # For the first step
                    probabilities = n2v_g.d_graph[walk[-1]][n2v_g.FIRST_TRAVEL_KEY]
                    next_node = int(np.random.choice(walk_options, size=1, p=probabilities)[0])
                else:
                    probabilities = n2v_g.d_graph[walk[-1]][n2v_g.PROBABILITIES_KEY][walk[-2]]
                    next_node = int(np.random.choice(walk_options, size=1, p=probabilities)[0])

                walk.append(next_node)

            else:
                break

        return walk

    def simulate_walk_original(self, walk_length, root, tmp_list):
        """
        simulating random walks from {root}.
        :param walk_length: the max length of a random walk.
        :param root: each random walk starts from {root}
        :param tmp_list: the first {tmp_list[node]} neighbors of {node} are
                         all within {self.until} hops from {root}.
                         suppose the last node of our walk is {cur},
                         we choose the next hop in {self.g[cur][:tmp_list[cur]]}
                         such that we will never walk out of {self.until} hops from root.
        :return: walk

        NOTE: this was the original version in RiWalk source code for generating the random walk
            the other version, apply biased random walk idea
        """
        walk = [root]
        g = self.g
        rand = np.random.randint(self.num_nodes, size=walk_length)
        for i in range(walk_length - 1):
            cur = walk[-1]
            cur_nbrs = g[cur]
            tl = tmp_list[cur]
            if tl:
                next_node = cur_nbrs[rand[i] % tl]
                walk.append(next_node)
            else:
                break

        return walk

    def simulate_walks_for_node(self, root, num_walks, walk_length, tmp_list):
        walks = []
        for walk_iter in range(num_walks):
            walk = self.simulate_walk(walk_length=walk_length, root=root, tmp_list=tmp_list)
            walks.append(walk)
        return walks

    def process_random_walks(self):
        vertices = np.random.permutation(self.num_nodes).tolist()
        chunks = partition(list(vertices), self.workers)
        futures = {}
        total_walk_time_ = 0
        total_bfs_time_ = 0
        total_ri_time_ = 0
        total_walks_writing_time_ = 0
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            part = 1
            for c in chunks:
                job = executor.submit(process_random_walks_chunk, self, c, part,
                                      self.until, self.num_walks, self.walk_length)
                futures[job] = part
                part += 1
            for job in as_completed(futures):
                walk_time_, bfs_time_, ri_time_, walk_writing_time_ = job.result()
                total_walk_time_ += walk_time_
                total_bfs_time_ += bfs_time_
                total_ri_time_ += ri_time_
                total_walks_writing_time_ += walk_writing_time_
        return total_walk_time_, total_bfs_time_, total_ri_time_, total_walks_writing_time_


def get_ri_walks(walks, root, ri_dict):
    ri_walks = []
    ri_dict[root] = root
    for walk in walks:
        ri_walk = [ri_dict[x] for x in walk]
        ri_walks.append(ri_walk)
    return ri_walks


def simple_log2(x):
    return x.bit_length() - 1


def floor_math_log2(x):
    return math.floor(math.log2(x))


def save_random_walks(walks, part, i):
    indexes = np.random.permutation(len(walks)).tolist()
    with open('walks/__random_walks_{}_{}.txt'.format(part, i), 'w') as f:
        for i in indexes:
            walk = walks[i]
            f.write(u"{}\n".format(u" ".join(str(v) for v in walk)))


def process_random_walks_chunk(ri_graph, vertices, part_id, until, num_walks, walk_length):
    walks_all = []
    i = 0
    tmp_list = [0] * ri_graph.num_nodes
    visited = [-1] * ri_graph.num_nodes
    wl_lists = [[0] * (until + 1) for _ in range(ri_graph.num_nodes)]
    bfs_time_ = 0
    walk_time_ = 0
    ri_time_ = 0
    walks_writing_time_ = 0
    for count, v in enumerate(vertices):
        bfs_begin_time_ = time.time()

        if 'na' == ri_graph.flag:
            if ri_graph.discount:
                n_attr = ri_graph.log_node_attr_
            else:
                n_attr = ri_graph.node_attr_

            node_layer_dict, node_attr_dict = ri_graph.bfs_na(v, tmp_list, visited, n_attr)
        else:
            node_layer_dict = ri_graph.bfs(v, tmp_list, visited)

        bfs_end_time_ = time.time()
        bfs_time_ += (bfs_end_time_ - bfs_begin_time_)

        nei_nodes = list(node_layer_dict.keys())

        # random walk generation
        walk_begin_time_ = time.time()
        walks = ri_graph.simulate_walks_for_node(v, num_walks, walk_length, tmp_list)
        walk_end_time_ = time.time()
        walk_time_ += (walk_end_time_ - walk_begin_time_)

        # generating riwalks
        ri_begin_time_ = time.time()
        if 'sp' == ri_graph.flag:
            sp_dict = ri_graph.get_sp_dict(v, node_layer_dict, nei_nodes)
            sp_walks = get_ri_walks(walks, v, sp_dict)
            walks_all.extend(sp_walks)

        if 'wl' == ri_graph.flag:
            wl_dict = ri_graph.get_wl_dict(v, node_layer_dict, nei_nodes, wl_lists, tmp_list)
            wl_walks = get_ri_walks(walks, v, wl_dict)
            walks_all.extend(wl_walks)

        if 'na' == ri_graph.flag:
            nsa_dict = ri_graph.get_na_dict(v, node_layer_dict, nei_nodes, node_attr_dict, wl_lists, tmp_list)
            nsa_walks = get_ri_walks(walks, v, nsa_dict)
            walks_all.extend(nsa_walks)

        if 'n2v' == ri_graph.flag:
            # no need to re-label the nodes
            walks_all.extend(walks)

        ri_end_time_ = time.time()
        ri_time_ += (ri_end_time_ - ri_begin_time_)

        if count % 100 == 0:
            logging.debug('worker {} has processed {} nodes.'.format(part_id, count))

        walks_writing_begin_time_ = time.time()
        if len(walks_all) > 1000000:
            save_random_walks(walks_all, part_id, i)
            i += 1
            walks_all = []
        walks_writing_end_time_ = time.time()
        walks_writing_time_ += (walks_writing_end_time_ - walks_writing_begin_time_)
    walks_writing_begin_time_ = time.time()
    save_random_walks(walks_all, part_id, i)
    walks_writing_end_time_ = time.time()
    walks_writing_time_ += (walks_writing_end_time_ - walks_writing_begin_time_)
    return walk_time_, bfs_time_, ri_time_, walks_writing_time_


# https://github.com/leoribeiro/struc2vec
def partition(lst, n):
    division = len(lst) / float(n)
    return [lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n)]
