import math
import torch
from model import Active_MNIST_BottomModel, Active_EMNIST_BottomModel, \
    Active_CCFD_BottomModel, Mnist_TopModel, EMNIST_TopModel, CCFD_TopModel, MNIST_BottomModel, EMNIST_BottomModel, \
    CCFD_BottomModel


class Active():
    '''
    The active party
    '''
    def __init__(self,data,data_nor,is_c,label,class_num,epsilon):
        self.data=data
        self.data_nor=data_nor
        self.label=label
        self.class_num=class_num
        self.epsilon=epsilon
        self.is_c=is_c


    def one_hot(self):
        '''
        one-hot encoding
        '''
        return torch.zeros(self.label.shape[0],self.class_num).scatter_(1,self.label.unsqueeze(1),1)

    def oue_perturb(self):
        '''
        perturb the label
        '''
        y_onehot=self.one_hot()
        p=float(1/2)
        q=float(1/(math.exp(self.epsilon)+1))
        for i in range(y_onehot.shape[0]):
            for j in range(y_onehot.shape[1]):
                value=y_onehot[i][j]
                if value==1:
                    if torch.rand(1)<p:
                        y_onehot[i][j]=1
                    else:
                        y_onehot[i][j]=0
                if value==0:
                    if torch.rand(1)<q:
                        y_onehot[i][j]=1
                    else:
                        y_onehot[i][j]=0

        return y_onehot

    def modify(self,y_perturb):
        y_zero=torch.zeros(y_perturb.shape)
        row_max=torch.argmax(y_perturb,dim=1)
        for i in range(row_max.shape[0]):
            y_zero[i][row_max[i]]=1
        return y_zero

    def build_bottom_model(self, input_dim, output_dim, hidden_dims, activation, dataset):
        if dataset =='mnist' or dataset == 'fmnist':
            return Active_MNIST_BottomModel(input_dim, output_dim, hidden_dims, activation=activation, is_top=False)
        elif dataset == 'emnist':
            return Active_EMNIST_BottomModel()
        elif dataset == 'ccfd2023':
            return Active_CCFD_BottomModel(input_dim, output_dim, hidden_dims, activation=activation, is_top=False)

    def build_top_model(self,input_dim, output_dims, hidden_dims, dataset):
        if dataset == 'mnist' or dataset == 'fmnist':
            return Mnist_TopModel(input_dim, output_dims, hidden_dims)
        elif dataset == 'emnist':
            return EMNIST_TopModel()
        elif dataset == 'ccfd2023':
            return CCFD_TopModel(input_dim, output_dims, hidden_dims, is_top=True)

    def aggregation(self,data,type='concatenate'):
        '''
        Aggregate intermediate results from passive parties
        '''
        if type=='concatenate':
            n=len(data)
            input_d=0
            agg_data=data[0]
            for i in range(n):
                input_d = input_d+data[i].shape[1]
                if i == 0:
                    continue
                else:
                    agg_data=torch.cat((agg_data,data[i]),1)
                return agg_data, input_d
        elif type == 'mean':
            input_d = data[0].shape[1]
            n = len(data)
            sum = torch.zeros(data[0].shape)
            for i in range(n):
                sum = sum+data[i]
            agg_data = sum/n
            return agg_data, input_d

class Passive():
    def __init__(self,data,data_nor,is_c):
        self.data=data
        self.data_nor=data_nor
        self.is_c=is_c


    def gini(self,y_onehot):
        num_total=torch.sum(y_onehot,dim=0)
        p_2=(num_total/len(y_onehot))**2
        gini_v=1-torch.sum(p_2,dim=0)
        return gini_v


    def continuous_gini_impurity(self,feature_values, y_onehot):

        sorted_indices = torch.argsort(feature_values)
        sorted_feature_values = torch.gather(feature_values, 0, sorted_indices)
        sorted_labels = y_onehot[sorted_indices]

#         best_split_point = None
#         min_gini = float('inf')
        indices = torch.nonzero(sorted_feature_values)
        if indices.numel()==0:
            return self.gini(sorted_labels)
#             best_split_point=len(sorted_labels)-1
        else:
            left_labels = sorted_labels[:indices[0][0],:]
            right_labels = sorted_labels[indices[0][0]:,:]
            return (len(left_labels) / len(y_onehot)) * self.gini(left_labels) + \
                     (len(right_labels) / len(y_onehot)) * self.gini(right_labels)



    def gini_impurity(self,y_onehot):
        gini_list=[]
        feature_num=self.data.shape[1]
        for i in range(feature_num):
            data_i=self.data[:,i]
            if self.is_c[i]:
                min_gini=self.continuous_gini_impurity(data_i,y_onehot)
                gini_list.append(min_gini)
            else:
                continue
        return gini_list
    def initialize(self,k,y_onehot):

        gini_list=self.gini_impurity(y_onehot)
        mu_list=[]
        for i in range(len(gini_list)):
            mu_list.append(k/gini_list[i])
        return gini_list,mu_list

    def build_model(self, input_dim, output_dim, hidden_dims, mu, activation, sigma, lam, dataset):
        if dataset == 'mnist' or dataset == 'fmnist':
            return MNIST_BottomModel(input_dim, output_dim, hidden_dims, mu=mu, sigma=sigma, lam=lam, activation=activation, is_top=False)
        elif dataset == 'emnist':
            return EMNIST_BottomModel(mu=mu, sigma=sigma, lam=lam)
        elif dataset == 'ccfd2023':
            return CCFD_BottomModel(input_dim, output_dim, hidden_dims, activation=activation, is_top=False)




