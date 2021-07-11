# Adjacency Matrix와 Stage Graph를 저장하는 방법으로는 일반 Numpy를 사용했습니다.
# 큰 데이터의 경우에는 이 부분을 scipy.sparse의 sparse matrix로 바꿀 필요가 있겠습니다.

import pandas as pd
import numpy as np
from dtw import *
from itertools import combinations
import time


class sunwoo_struc2vec_training():

    def __init__(self, data):
        """
        Adjacency matrix is getting input!
        For the bigger graph, it would be better to get edge index as an input
        :param data: numpy type adjacency matrix
        """
        self.data = data
        self.num_node = data.shape[0]

    def initialize(self, k):
        # Initializing Vectors
        self.h = np.random.normal(0, 0.1, size = (self.num_node, k))
        self.v = np.random.normal(0, 0.1, size=(self.num_node, k))

    def generating_similarity(self, K):

        self.stage_num = K
        node_edge = np.sum(self.data, 0)
        total_adj = self.data
        # 각 Hop의 Adjacency들을 쌓아줍니다.
        for k in range(K):
            if k == 0:
                next_step = self.data.dot(self.data)
                np.fill_diagonal(next_step, 0)
                total_adj = np.vstack([[total_adj], [next_step]])  # Dim 맞추기
            else:
                next_step = next_step.dot(self.data)
                np.fill_diagonal(next_step, 0)
                total_adj = np.vstack([total_adj, [next_step]])  # Dim 맞추기

        # 먼저 모든 node의 R()을 계산합니다.

        hop_neighbors = []  # 여기는 이제 Neighbors가 쌓입니다.
        hop_sequences = []  # 여기는 이제 Sequence가 쌓입니다!
        for i in range(self.num_node):
            entire_root = []
            part_neighbors = []
            part_sequence = []
            for cur_hop in range(K):
                row_extracted = total_adj[cur_hop][i]
                indexes = np.where(row_extracted > 0)[0]  # 값이 있는 데이터만 추출
                to_be_stacked = indexes[np.invert(np.isin(indexes, entire_root))]  # 사전 hop에 있었다면 stacking X
                entire_root += list(to_be_stacked)
                part_neighbors.append(list(to_be_stacked))
                part_sequence.append(list(node_edge[to_be_stacked]))

            hop_neighbors.append(part_neighbors)
            hop_sequences.append(part_sequence)

        self.hop_neighbors = hop_neighbors
        self.sequences = hop_sequences
        #######

        # 메모리 효율을 위해 사전에 추출했던 total_adj를 재활용합니다!
        for hop in range(K):
            total_adj[hop] = np.zeros((self.num_node, self.num_node))
            comb = combinations(np.arange(self.num_node), 2)  # Full Network 를 위해!
            for choice in comb:
                if hop == 0:
                    g_x1_x2 = dtw(self.sequences[choice[0]][hop],
                                  self.sequences[choice[1]][hop]).distance  # Dynamic Time Warping
                    total_adj[hop][choice] = g_x1_x2
                else:
                    g_x1_x2 = dtw(self.sequences[choice[0]][hop],
                                  self.sequences[choice[1]][hop]).distance  # Dynamic Time Warping
                    total_adj[hop][choice] = total_adj[hop - 1][choice] + g_x1_x2  # Monotone Increasing!

            total_adj[hop] += total_adj[hop].T

        f = lambda temp: np.exp(-temp)
        total_adj = f(total_adj)
        for hop in range(K): np.fill_diagonal(total_adj[hop], 0)
        self.weight_matrix = total_adj[:-1]

        ## 이제 Weight Matrix에 Weight가 모두 쌓인 상태.
        ## 이어서 각자 층 별로 왔다갔다 할 수 있는 확률을 정의해주자! 리스트 형식으로 쌓을것이다.

        moving_up_down = []
        for hop in range(K):
            hop_mean = np.mean(self.weight_matrix[hop])
            partial_hop_moving = []
            for node_ in range(self.num_node):
                partial_node_moving = []
                if hop == 0:
                    partial_node_moving.append([0, 1])
                elif hop == K - 1:
                    partial_node_moving.append([1, 0])
                else:  # 위 혹은 아래로 이동해야한다.
                    partial_node_moving.append([np.sum(self.weight_matrix[hop][node_] > hop_mean), 1])
                partial_hop_moving.append(partial_node_moving)
            moving_up_down.append(partial_hop_moving)

        self.itself_moving = moving_up_down

        # self.weight에 다른 node로 in-stage에서 Moving할 확률이 할당됨
        # self.itself_moving에 Stage간 Moving의 확률이 할당됨 (Up/Down)
        # 근데 self.itself_moving은 2겹으로 싸였으니까 주의!
        print("Weight Matrix 등 학습 준비가 완료되었습니다!")

    def build_tree(self) :

        # Huffman Binary Tree를 구성했습니다.

        edge_list = pd.DataFrame({'node': np.arange(self.num_node), 'edge': np.sum(self.data, 1)})
        huffman_info = [[i, [], []] for i in np.arange(self.num_node)]
        sorted_df = edge_list.sort_values('edge')
        tree_info = []
        for i, j in zip(sorted_df.node.values, sorted_df.edge.values):
            tree_info.append([j, i, []])

        count = 0

        while len(tree_info) > 1:

            tree_info = sorted(tree_info)
            hidden_unit = self.num_node - 2 - count  # Node Num  - i
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

    def random_selector (self, condition, prob_seq) : # 간혹 확률의 Summation이 1이 되지 않아 에러가 뜨는것을 방지해준다.

        if condition == 0 : # Stage 간 Moving
            p_ = np.array(prob_seq)/np.sum(prob_seq)
            pos = np.where(p_==np.max(p_))[0][0]
            if np.sum(p_) > 1 :
                p_[pos] -= np.sum(p_) - 1
            elif np.sum(p_) < 1 :
                p_[pos] += 1-np.sum(p_)
            next_node = np.random.choice([-1,1], p = p_)
        else : # Stage 내 Node 이동
            p_ = prob_seq/np.sum(prob_seq)
            pos = np.where(p_==np.max(p_))[0][0]
            if np.sum(p_) > 1 :
                p_[pos] -= np.sum(p_) - 1
            elif np.sum(p_) < 1 :
                p_[pos] += 1-np.sum(p_)
            next_node = np.random.choice(np.arange(self.num_node), p = p_)

        return next_node

    def generate_random_walk(self, cur_node, walk_length, q = 0.5) :

        # self.itself_moving이 위아래 왔다갔다
        # self.weight_matrix가 Stage 안에서 왔다갔다
        sequence = []
        position, length = int(cur_node), int(walk_length)
        sequence.append(position)
        cur_pos = position  # Indicates the Current Position
        cur_stage = 0
        for i in range(walk_length-1) :
            decision = np.random.choice(['leave', 'stay'], p = [1-q, q])
            while decision== 'leave' :
                probs = self.itself_moving[cur_stage][cur_pos][0]
                where_to_go = self.random_selector(0, probs)
                cur_stage += where_to_go
                decision = np.random.choice(['leave', 'stay'], p=[1 - q, q])

            next_node = self.random_selector(1, self.weight_matrix[cur_stage][cur_pos])
            sequence.append(next_node) ; cur_pos = next_node

        return sequence

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

    def get_ready(self, K) :
        """
        This MUST be implement before the fit function
        :param K: Stage Numbers
        :return:
        """
        self.build_tree()
        self.generating_similarity(K)


    def fit(self, epochs, learning_rate, walk_length, window_size, hidden_size):

        start_time = time.time()
        self.learning_rate = learning_rate
        self.epochs = epochs  # also known as gamma
        self.walk_length = walk_length
        self.window_size = window_size

        if self.walk_length < 2 * self.window_size:
            raise TypeError('Window Size is smaller than walk length. Set the longer walk length')

        self.initialize(hidden_size)
        print('학습이 시작됩니다! Go~')
        for epoch in range(self.epochs):
            order = np.arange(self.num_node)
            self.order = order
            np.random.shuffle(order)

            for curr_target in order:
                training_sequence = self.generate_random_walk(curr_target, self.walk_length)

                for per_walk in range(len(training_sequence)):
                    x_index, y_output = self.sequence_neighbor_extractor(training_sequence, self.window_size,
                                                                         per_walk)
                    self.train(x_index, y_output)

        print('Learning Finished! It took {} sec'.format(time.time() - start_time))