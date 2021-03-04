import numpy as np
import pandas as pd
import math
import time

class sunwoo_huffman_hs () :

    def __init__(self, data) :

        self.n_nodes = data.shape[0]
        self.height = int(math.trunc(np.log2(self.n_nodes - 1)) + 1) - 1  # For the Python Dimension = number_of_nodes
        self.data = data

    def build_tree(self) :

        edge_list = pd.DataFrame({'node': np.arange(34), 'edge': np.sum(self.data, 1)})
        huffman_info = [[i, [], []] for i in np.arange(self.n_nodes)]
        sorted_df = edge_list.sort_values('edge')
        tree_info = []
        for i, j in zip(sorted_df.node.values, sorted_df.edge.values):
            tree_info.append([j, i, []])

        count = 0

        while len(tree_info) > 1:

            tree_info = sorted(tree_info)
            hidden_unit = 34 - 2 - count  # Node Num  - i
            left_child = tree_info[0]
            right_child = tree_info[1]

            # 왼쪽 자식노드와 오른쪽 자식노드가 모두 Leaf 노드인 경우
            if (left_child[2] == []) and (right_child[2] == []):
                huffman_info[left_child[1]][1].append(hidden_unit)
                huffman_info[left_child[1]][2].append(1)
                huffman_info[right_child[1]][1].append(hidden_unit)
                huffman_info[right_child[1]][2].append(2)
                new_hidden = [left_child[0] + right_child[0], hidden_unit, []]
                new_hidden[2].append(left_child[1]);
                new_hidden[2].append(right_child[1])
                tree_info.pop(0);
                tree_info.pop(0)
                tree_info.append(new_hidden)

            # 왼쪽 자식노드는 Leaf고 오른쪽 자식노드는 Inner Unit
            elif (right_child[2] != []) and (left_child[2] == []):  # 왼쪽만 Leaf인 경우
                new_hidden = [left_child[0] + right_child[0], hidden_unit, []]
                huffman_info[left_child[1]][1].append(hidden_unit)
                huffman_info[left_child[1]][2].append(1)
                for small_sample1 in right_child[2]:
                    huffman_info[small_sample1][1].insert(0, hidden_unit)
                    huffman_info[small_sample1][2].insert(0, 2)
                    new_hidden[2].append(small_sample1)
                new_hidden[2].append(left_child[1])
                tree_info.pop(0);
                tree_info.pop(0)
                tree_info.append(new_hidden)

            # 왼쪽 자식노드는 Inner Unit 이고 오른쪽 자식노드는 Leaf Unit
            elif (right_child[2] == []) and (left_child[2] != []):  # 오른쪽만 leaf인 경우
                new_hidden = [left_child[0] + right_child[0], hidden_unit, []]
                huffman_info[right_child[1]][1].append(hidden_unit)
                huffman_info[right_child[1]][2].append(2)
                for small_sample2 in left_child[2]:
                    huffman_info[small_sample2][1].insert(0, hidden_unit)
                    huffman_info[small_sample2][2].insert(0, 1)
                    new_hidden[2].append(small_sample2)
                new_hidden[2].append(right_child[1])
                tree_info.pop(0);
                tree_info.pop(0)
                tree_info.append(new_hidden)

            else:  # 둘 다 Hidden인 경우
                new_hidden = [left_child[0] + right_child[0], hidden_unit, []]
                for small_sample3 in left_child[2]:
                    huffman_info[small_sample3][1].insert(0, hidden_unit)
                    huffman_info[small_sample3][2].insert(0, 1)
                    new_hidden[2].append(small_sample3)
                for small_sample4 in right_child[2]:
                    huffman_info[small_sample4][1].insert(0, hidden_unit)
                    huffman_info[small_sample4][2].insert(0, 2)
                    new_hidden[2].append(small_sample4)
                tree_info.pop(0);
                tree_info.pop(0)
                tree_info.append(new_hidden)

            count += 1

            self.huffman_node_info = huffman_info

    def initializing(self, embedding_dimension, method, var):

        self.e_dim = embedding_dimension
        if method == 'uniform':
            self.h = np.random.uniform(0, var, self.n_nodes * self.e_dim).reshape(int(self.n_nodes),
                                                                                  int(self.e_dim))
            self.v = np.random.uniform(0, var, (self.n_nodes - 1) * self.e_dim).reshape((int(self.n_nodes) - 1),
                                                                                        int(self.e_dim))
        elif method == 'normal':
            self.h = np.random.normal(0, var, self.n_nodes * self.e_dim).reshape(int(self.n_nodes), int(self.e_dim))
            self.v = np.random.normal(0, var, (self.n_nodes - 1) * self.e_dim).reshape((int(self.n_nodes) - 1),
                                                                                       int(self.e_dim))
        else:
            raise TypeError('Either uniform or normal should be passed')

    def generate_random_walk(self, index_, walk_length):
        """
        :param index_: the target node
        :param walk_length: Length of random walk - t
        :return: sequence of random walk (list)
        """

        sequence_ = []
        position, length = int(index_), int(walk_length)
        sequence_.append(position)
        cur_pos = position  # Indicates the Current Position

        for rw_loop in range(1, length):
            cur_pos = np.random.choice(np.where(self.data[cur_pos])[0], 1)[0]
            sequence_.append(cur_pos)

        return sequence_

    def sequence_neighbor_extractor(self, sequence1, window_size, target_position):

        """
        :param sequence1: Sequence Generated by
        :param window_size: as it is written
        :param target_position: index that would be a target
        :return: required index for the model
        """

        input_index = sequence1[target_position]
        target_index = []

        if target_position < window_size:
            target_index = sequence1[:target_position] + sequence1[
                                                         target_position + 1: target_position + window_size + 1]

        elif len(sequence1[target_position + 1:]) < window_size:
            target_index = sequence1[target_position - window_size: target_position] + sequence1[target_position + 1:]


        else:
            target_index = sequence1[target_position - window_size: target_position] + sequence1[ target_position + 1: target_position + window_size + 1]

        return input_index, target_index

    def sigmoid(self, x):
        res = 1 / (1 + np.exp(-x))
        return res

    def train(self, x_index, y_list):
        in_h = self.h[x_index, :]
        for y_index in y_list:
            hidden_list, target_path = self.huffman_node_info[y_index][1], self.huffman_node_info[y_index][2]
            v_vec = self.v[hidden_list, :].copy()
            prob = self.sigmoid(v_vec.dot(in_h.reshape(-1, 1)))  # V x 1
            true_v = np.array(target_path).reshape(len(target_path), 1) - 2
            self.v[hidden_list, :] = v_vec - self.learning_rate * (prob + true_v).dot(in_h.reshape(1, -1))
            self.h[x_index, :] = in_h - self.learning_rate * np.sum(((prob + true_v) * v_vec), 0)

    def fit(self, epochs, learning_rate, walk_length, window_size, hidden_size, init_method, var):

        start_time = time.time()
        self.learning_rate = learning_rate
        self.epochs = epochs  # also known as gamma
        self.walk_length = walk_length
        self.window_size = window_size

        if self.walk_length < 2 * self.window_size:
            raise TypeError('Window Size is smaller than walk length. Set the longer walk length')

        self.initializing(hidden_size, init_method, var)
        print('학습이 시작됩니다! Go~')
        for epoch in range(self.epochs):
            order = np.arange(self.n_nodes)
            self.order = order
            np.random.shuffle(order)

            for curr_target in order:
                training_sequence = self.generate_random_walk(curr_target, self.walk_length)

                for per_walk in range(len(training_sequence)):
                    x_index, y_output = self.sequence_neighbor_extractor(training_sequence, self.window_size,
                                                                         per_walk)

                    self.train(x_index, y_output)

        print('Learning Finished! It took {} sec'.format(time.time() - start_time))