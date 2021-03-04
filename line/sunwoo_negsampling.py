import numpy as np
import time

class LINE_multiple :

    def __init__(self,  data) :

        self.data = data
        self.num_node = data.shape[0]
        self.sampling_data = data.copy()
        for i in range(self.num_node) : self.sampling_data[i,i] = 1

    def negative_sampler(self, cur_index, k, power) :

        prob_list = np.sum(self.data, 0) # 각 노드별로 Summation
        sampled_cand = np.where(self.sampling_data[cur_index] < 1)[0]  # sampling_data는 diag 값을 1로 했음!
        each_prob = (prob_list[sampled_cand]**power)/np.sum(prob_list[sampled_cand]**power)

        # 간혹 이 확률이 반올림의 문제로 1이 되지 않는 경우가 발생
        if np.sum(each_prob) > 1 :
            each_prob[0] = each_prob[0] - (np.sum(each_prob) - 1) # 넘치는만큼 빼주자!

        elif np.sum(each_prob) < 1 :
            each_prob[0] = each_prob[0] + (1 - np.sum(each_prob)) # 모자란만큼 더해주자!

        return np.random.choice(sampled_cand, k, p = each_prob, replace = False)

    def sigmoid(self, x):

        return 1/(1+np.exp(-x))

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

    def fit (self, condition, training_style, k, power, h, init_method, var, lr, epochs) :

        self.initialize(condition, h, init_method, var)
        start_time = time.time()
        print('Learning Started!')
        for_visualizing = []

        if training_style == 'basic' :

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
                    for_visualizing.append(self.u.copy())
                self.loss = total_loss
                print('Learning Finished! It took {} seconds'.format(time.time() - start_time))
                self.result = for_visualizing

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
                    for_visualizing.append(self.u.copy())
                    total_loss.append(np.mean(loss_part))
                self.loss = total_loss
                print('Learning Finished! It took {} seconds'.format(time.time() - start_time))
                self.result = for_visualizing

        elif training_style == 'neg_sampling' :

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
                        neg_samples = self.negative_sampler(i, k, power)
                        u_neg = self.u[neg_samples]
                        loss = -np.log(self.sigmoid(ui*uj)) - np.sum(np.log(self.sigmoid(np.sum(-ui*u_neg, 1))))

                        # 3개의 Update 구성
                        new_uj = uj - lr * (self.sigmoid(np.sum(ui * uj)) - 1) * ui
                        new_neg = u_neg - lr * (self.sigmoid(np.sum(ui * u_neg, 1).reshape(-1,1))).dot(ui.reshape(1,-1))
                        new_ui = ui - lr * (self.sigmoid(np.sum(ui * uj)) - 1) * uj - lr * np.sum((self.sigmoid(np.sum(u_neg*ui, 1)).reshape(-1,1))*u_neg, 0)
                        partial_loss.append(loss)

                        # Update 실행!
                        self.u[i] = new_ui
                        self.u[j] = new_uj
                        self.u[neg_samples] = new_neg
                    total_loss.append(np.mean(partial_loss))
                    for_visualizing.append(self.u.copy())
                self.loss = total_loss
                print('Learning Finished! It took {} seconds'.format(time.time() - start_time))
                self.result = for_visualizing

            elif condition == 'sop' : # Updating Negative Sampling Second-Order Proximity

                cor = np.where(self.data > 0)
                edge_num = np.where(self.data > 0)[0].shape[0]
                index = np.arange(edge_num)
                total_loss = []
                for epoch in np.arange(epochs):
                    np.random.shuffle(index)
                    partial_loss = []
                    loss_part = []
                    for ind in index:
                        i, j = cor[0][ind], cor[1][ind]  # i is given index / j is Context
                        ui, cj = self.u[i], self.c[j]
                        neg_samples = self.negative_sampler(i, k, power)
                        c_neg = self.c[neg_samples]
                        loss = -np.log(self.sigmoid(ui * cj)) - np.sum(np.log(self.sigmoid(np.sum(-ui * c_neg, 1))))

                        # 3개의 Update 구성
                        new_cj = cj - lr * (self.sigmoid(np.sum(ui * cj)) - 1) * ui
                        new_neg = c_neg - lr * (self.sigmoid(np.sum(ui * c_neg, 1).reshape(-1, 1))).dot(ui.reshape(1, -1))
                        new_ui = ui - lr * (self.sigmoid(np.sum(ui * cj)) - 1) * cj - lr * np.sum((self.sigmoid(np.sum(c_neg * ui, 1)).reshape(-1, 1)) * c_neg, 0)

                        partial_loss.append(loss)

                        self.u[i] = new_ui
                        self.c[j] = new_cj
                        self.c[neg_samples] = new_neg
                    total_loss.append(np.mean(partial_loss))
                    for_visualizing.append(self.u.copy())
                    self.loss = total_loss
                print('Learning Finished! It took {} seconds'.format(time.time() - start_time))
                self.result = for_visualizing

