import numpy as np

class LINE () :

    def __init__(self,  data) :

        self.data = data
        self.num_node = data.shape[0]

    def initialize(self, condition, h, init_method, var):

        # Choosing the Embedding Dimension and Method
        self.h = h
        if init_method == 'normal' :
            self.u = np.random.normal(0, var, self.num_node*self.h).reshape(self.num_node, self.h)
            if condition == 'sop' :
                self.c = np.random.normal(0, var, self.num_node*self.h).reshape(self.num_node, self.h)
        elif init_method == 'uniform' :
            self.u = np.random.uniform(0, var, self.num_node * self.h).reshape(self.num_node, self.h)
            if condition == 'sop' :
                self.c = np.random.uniform(0, var, self.num_node * self.h).reshape(self.num_node, self.h)
        else :
            raise TypeError("init_method should be given either normal or uniform should be given")

        if condition != 'fop' :
            if condition != 'sop' :
                raise TypeError('Condition should be given either fop or sop')

    def fit (self, condition, h, init_method, var, lr, epochs) :

        self.initialize(condition, h, init_method, var)
        print('Learning Started!')

        if condition == 'fop' :  # Updating First-Order Proximity
            cor = np.where(self.data > 0)
            edge_num = np.where(self.data > 0)[0].shape[0]
            index = np.arange(edge_num)
            total_loss = []
            for epoch in np.arange(epochs) :
                np.random.shuffle(index)
                partial_loss = []
                for ind in index : # i is given / j in non-given
                    i, j = cor[0][ind], cor[1][ind]
                    ui, uj = self.u[i], self.u[j]
                    denom = np.sum(np.exp(np.sum(self.u*ui, 0)))
                    p1 = np.exp(np.sum(ui*uj))/denom
                    self.u[i] = ui + lr * (uj - np.sum(self.u*np.exp(np.sum(self.u*ui, 1).reshape(-1, 1)), 0)/denom )
                    self.u[j] = uj + lr * (ui - ui*np.exp(np.sum(ui*uj))/denom)
                    loss = -np.log(p1)
                    partial_loss.append(loss)
                total_loss.append(np.mean(partial_loss))
            self.loss = total_loss
            print('Learning Finished!')

        elif condition == 'sop' :  # i is given (embedding V) / j in non-given (context)
            cor = np.where(self.data > 0)
            edge_num = np.where(self.data > 0)[0].shape[0]
            index = np.arange(edge_num)
            total_loss = []
            for epoch in np.arange(epochs) :
                np.random.shuffle(index)
                loss_part = []
                for ind in index :
                    i , j = cor[0][ind] , cor[1][ind]  # i is given index / j is Context
                    denominator = np.sum(np.exp(np.sum(self.u[i]*(self.c), 1)))
                    p2 = np.exp(np.sum(self.u[i] * self.c[j]))/denominator
                    loss = -np.log(p2)

                    # Updating Embedding Vector
                    new_u = self.u[i] + lr*(self.c[j] - ((np.sum(self.c*np.exp((np.sum(self.u[i]*(self.c), 1).reshape(-1,1))), 0))/denominator) )

                    # Updating Context Vector
                    dE_dc = np.exp(np.sum(self.u[i]*self.c, 1).reshape(-1 , 1)).dot(self.u[i].reshape(1, -1))/denominator
                    dE_dc[j] = dE_dc[j] - self.u[i]
                    self.c = self.c - lr*dE_dc
                    self.u[i] = new_u

                    loss_part.append(loss)
                total_loss.append(np.mean(loss_part))
            self.loss = total_loss
            print('Learning Finished!')