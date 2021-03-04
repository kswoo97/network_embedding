import numpy as np
import time

class deep_walk() :
    """
    CSE week 4 code Submission
    By App.Stat sunwoo Kim
    """

    def __init__ (self, data) :

        """
        :param data: Adjacency Matrix of Data
        """

        self.data = data
        self.num_node = data.shape[0]

    def initializer (self, hidden_size, method) :
        """
        :param hidden_size: Determines the size of Embedding Space
        method : 'uniform' : uniform distribution [0,1]
        method : 'normal' : standard normal distribution
        :return: W , W_
        """
        # W : V x N / W_ : N x V
        self.hidden_size = hidden_size
        if method == 'uniform' :
            self.W = np.random.uniform(0,1, (self.hidden_size * self.num_node)).reshape(self.num_node, self.hidden_size)
            self.W_ = np.random.uniform(0,1, (self.hidden_size * self.num_node)).reshape(self.hidden_size, self.num_node)
        elif method == 'normal' :
            self.W = np.random.normal(0,1, (self.hidden_size * self.num_node)).reshape(self.num_node, self.hidden_size)
            self.W_ = np.random.normal(0, 1, (self.hidden_size * self.num_node)).reshape(self.hidden_size, self.num_node)
        else :
            raise NameError('method should be either uniform or normal')

    def generate_random_walk(self, index_, walk_length) :
        """
        :param index_: the target node
        :param walk_length: Length of random walk - t
        :return: sequence of random walk (list)
        """

        sequence_ = []
        position, length = int(index_), int(walk_length)
        sequence_.append(position)
        cur_pos = position # Indicates the Current Position

        for rw_loop in range(1, length) :
            cur_pos = np.random.choice(np.where(self.data[cur_pos])[0], 1)[0]
            sequence_.append(cur_pos)

        return sequence_

    def sequence_neighbor_extractor(self, sequence1, window_size, target_position) :

        """
        :param sequence1: Sequence Generated by
        :param window_size: as it is written
        :param target_position: index that would be a target
        :return: required index for the model
        """

        input_index = sequence1[target_position]
        target_index = []


        if target_position < window_size :
            target_index = sequence1[:target_position] + sequence1[target_position + 1 : target_position + window_size + 1]

        elif len(sequence1[target_position + 1 : ]) < window_size:
            target_index = sequence1[target_position - window_size : target_position] + sequence1[target_position + 1 : ]

        else :
            target_index = sequence1[target_position - window_size : target_position] + sequence1[target_position + 1 : target_position + window_size + 1]

        return input_index, target_index

    def one_hot_encoding(self, index_2) :

        """
        :param index_2: position where to be filled with 1
        :return: one hot encoded vector with shape V x 1
        """
        enc_vec = np.zeros(self.num_node).reshape(self.num_node, 1)
        enc_vec[index_2] = 1
        return enc_vec

    def softmax(self, array_) :
        """
        :param array_: receives V x 1 array
        :return: array like probability
        """
        prob_array = np.exp(array_)/np.sum(np.exp(array_))
        return prob_array

    def cross_entropy(self, pred, target_index, window_size):

        E = -np.sum(pred[target_index]) + window_size*np.log(np.sum(pred))

        return E

    def forward_prop(self, window_size, x_index, target_index) :
        """
        :param window_size: as it is written. window size
        :param x_index: the input index of the list / int
        :param target_index: the output index of the list / list
        :return: TBD
        """
        self.x_index = x_index
        self.window_size = window_size
        train_x = self.one_hot_encoding(x_index)
        for k in target_index :
            if train_x.ndim == 2 :
                train_x = np.vstack([[train_x], [self.one_hot_encoding(k)]])
            elif train_x.ndim == 3 :
                train_x = np.vstack([train_x, [self.one_hot_encoding(k)]])
        self.new_ = train_x
        self.train_y = train_x[1:]  # Arrays are stacked in this value
        self.train_x = train_x[0]
        self.h = self.W.transpose().dot(self.train_x)  # N x 1 ; where N indicates the hidden size

        self.u = self.W_.transpose().dot(self.h)  # V x 1 ; final values
        self.y = self.softmax(self.u)
        error = self.cross_entropy(self.y, target_index, self.window_size)

        return error

    def backward_prop(self) :

        # Just Operate it, then weights are updated itself

        # Calculating the Latter Weight
        EI = self.train_y.shape[0]*self.y - np.sum(self.train_y, axis = 0) # V x 1 Shape
        self.EI = EI
        grad_ =  self.h.dot(EI.reshape(1, EI.shape[0]))
        new_W_ = self.W_ - self.learning_rate*grad_

        # Calculating & Updating the Former Weight
        grad = np.sum((self.W_ * EI.transpose()), axis = 1)  # N x V Shape
        self.W[self.x_index, :] = self.W[self.x_index, :] - self.learning_rate*grad

        # Updating the Latter Weight
        self.W_ = new_W_

    def fit(self, epochs, learning_rate, walk_length, window_size, hidden_size, init_method) :
        """
        :param epochs: as it is written / written as gamma in the paper
        :param learning_rate: leraning rate
        :param walk_length: length of the sequence / written as t in the paper
        :param window_size: window size / written as w in the paper
        :return: Nothing. Training is Happening
        """
        start_time = time.time()
        self.learning_rate = learning_rate
        self.epochs = epochs # also known as gamma
        self.walk_length = walk_length
        self.window_size = window_size

        if self.walk_length < 2*self.window_size :
            raise TypeError('Window Size is smaller than walk length. Set the longer walk length')

        self.initializer(hidden_size, init_method)
        loss_plot_list = []

        print('Learning Started!')
        for epoch in range(self.epochs) : # Epoch Starting
            order = np.arange(self.num_node)
            self.order = order
            np.random.shuffle(order)

            for curr_target in order :
                training_sequence = self.generate_random_walk(curr_target, self.walk_length)

                for per_walk in range(len(training_sequence)) :
                    x_index, y_output = self.sequence_neighbor_extractor(training_sequence, self.window_size, per_walk)
                    loss = self.forward_prop(self.window_size, x_index, y_output) ## forward prop
                    self.backward_prop() # backprop

            loss_plot_list.append(loss)

        print('Learning Finished! It took {} sec'.format(time.time() - start_time))

if __name__ == '__main__' :

    # Reading Data

    path = "C:\\Users\\kswoo\\OneDrive\\바탕 화면\\cse_ur\\week4\\"
    a = []
    with open(path + 'karate_club.adjlist') as f:
        for line in f :
            a.append(line.rstrip().split())
    a = a[3:]

    adj_array = np.zeros(len(a) * len(a)).reshape(len(a), len(a))
    for i1 in a:
        for i2 in range(len(i1)):
            index1 = int(i1[0]);
            index2 = int(i1[i2])
            adj_array[index1, index2] = 1

    adj_array = adj_array + adj_array.transpose()
    for i in range(adj_array.shape[0]): adj_array[i, i] = 0 # Deleting the Diagonal Terms

    # Now Adjacency matrix had been assigned to 'adj_array'

    model = deep_walk(adj_array)
    model.fit(epochs = 5, learning_rate = 0.02, walk_length = 10, window_size = 3, hidden_size = 2, init_method= 'uniform')
    print(model.W)