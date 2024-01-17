from math import floor
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torchvision import datasets
from utils import split_data, clip, gaussion_noise, getrandomk_lap
from torchvision import transforms
from participant import Active,Passive
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from options import args_parser
from lap_score import construct_W,lap_score
import openpyxl

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 20, 't': 1}
    parsed = args_parser()

    # Load data
    transform1 = transforms.Compose([transforms.ToTensor(),
                              ])
    transform2 = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,)),
                              ])
    transform3 = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

    dataset = parsed['dataset']
    if dataset == 'mnist':
        # raw data
        raw_train = datasets.MNIST('mnist', download=True, train=True, transform=transform1)
        raw_val = datasets.MNIST('mnist', download=True, train=False, transform=transform1)
        # standardized data
        trainset = datasets.MNIST('mnist', download=True, train=True, transform=transform2)
        valset = datasets.MNIST('mnist', download=True, train=False, transform=transform2)
        # label
        labels = datasets.MNIST("mnist", download=True, train=True).targets[:30000]  # don't need all to demonstrate value
        labels_val = datasets.MNIST("mnist", download=True, train=False).targets[:5000]

    elif dataset == 'fmnist':
        raw_train = datasets.FashionMNIST('fm', download=True, train=True, transform=transform1)
        raw_val = datasets.FashionMNIST('fm', download=True, train=False, transform=transform1)

        trainset = datasets.FashionMNIST('fm', download=True, train=True, transform=transform2)
        valset = datasets.FashionMNIST('fm', download=True, train=False, transform=transform2)

        labels = datasets.FashionMNIST("fm", download=True, train=True).targets[:30000]  # don't need all to demonstrate value
        labels_val = datasets.FashionMNIST("fm", download=True, train=False).targets[:5000]

    elif dataset == 'emnist':
        raw_train = datasets.EMNIST(root='em', train=True, transform=transform1, download=True, split='letters')
        raw_val = datasets.EMNIST(root='em', train=False, transform=transform1, download=True, split='letters')
        # # 标准化后的数据
        trainset = datasets.EMNIST(root='em', train=True, transform=transform3, download=True, split='letters')
        valset = datasets.EMNIST(root='em', train=False, transform=transform3, download=True, split='letters')

        labels = datasets.EMNIST(root='./data', download=True, train=True, split='letters').targets[:124800] - 1  # don't need all to demonstrate value
        labels_val = datasets.EMNIST(root='./data', download=True, train=False, split='letters').targets[:20800] - 1
    elif dataset == 'ccfd':
        df = pd.read_csv("creditcard_2023.csv")
        df = df.drop('id', axis=1)
        SC = StandardScaler()
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        X = SC.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        labels = torch.tensor(y_train)
        labels_val = torch.tensor(y_test)
    '''
     split data
    '''
    if dataset == 'ccfd':
        train_1 = X_train[:, :15]
        train_2 = X_train[:, 15:]
        test_1 = X_test[:, :15]
        test_2 = X_test[:, 15:]
    elif dataset == 'emnist':
        raw_img, _, _ = split_data(raw_train)
        stan_img, _, _ = split_data(trainset)

        raw_train_1 = torch.cat(raw_img[0][:124800])
        raw_train_1 = raw_train_1.view(raw_train_1.shape[0], -1)

        raw_train_2 = torch.cat(raw_img[1][:124800])
        raw_train_2 = raw_train_2.view(raw_train_2.shape[0], -1)

        train_1 = torch.cat(stan_img[0][:124800])
        train_1 = train_1.view(train_1.shape[0], -1)

        train_2 = torch.cat(stan_img[1][:124800])
        train_2 = train_2.view(train_2.shape[0], -1)

        val_img, _, _ = split_data(valset)
        test_1 = torch.cat(val_img[0][:20800])
        test_1 = test_1.view(test_1.shape[0], -1)

        test_2 = torch.cat(val_img[1][:20800])
        test_2 = test_2.view(test_2.shape[0], -1)
    else:
        raw_img, _, _ = split_data(raw_train)
        stan_img, _, _ =split_data(trainset)

        # raw data of passive party
        raw_train_1= torch.cat(raw_img[0][:30000])
        raw_train_1=raw_train_1.view(raw_train_1.shape[0], -1)
        # raw data of active party
        raw_train_2= torch.cat(raw_img[1][:30000])
        raw_train_2=raw_train_2.view(raw_train_2.shape[0], -1)

        train_1= torch.cat(stan_img[0][:30000])
        train_1=train_1.view(train_1.shape[0], -1)

        train_2= torch.cat(stan_img[1][:30000])
        train_2=train_2.view(train_2.shape[0], -1)


        val_img, _, _= split_data(valset)
        test_1 = torch.cat(val_img[0][:5000])
        test_1 = test_1.view(test_1.shape[0], -1)

        test_2 = torch.cat(val_img[1][:5000])
        test_2 = test_2.view(test_2.shape[0], -1)

    if dataset == 'ccfd':
        is_cf = torch.ones(train_2.shape[1], dtype=torch.bool)
        partyB = Active(train_2, train_2, is_cf, labels, parsed['num_classes'], parsed['epsilon'])

        y_onehot_p = partyB.oue_perturb()
        y_onehot = partyB.modify(y_onehot_p)

        is_cf_A = torch.ones(train_1.shape[1], dtype=torch.bool)
        partyA = Passive(train_1, train_1, is_cf_A)

        ginilist_A, mulist_A = partyA.initialize(0.5, y_onehot)
        mulist_B = torch.tensor(mulist_A)
    else:
        is_cf = torch.ones(raw_train_2.shape[1],dtype=torch.bool)
        partyB = Active(raw_train_2, train_2, is_cf, labels, parsed['num_classes'],parsed['epsilon'])

        y_onehot_p = partyB.oue_perturb()
        y_onehot = partyB.modify(y_onehot_p)

        is_cf_A = torch.ones(raw_train_1.shape[1],dtype=torch.bool)
        partyA = Passive(raw_train_1,train_1,is_cf_A)

        ginilist_A, mulist_A = partyA.initialize(parsed['coefficient'], y_onehot)

    # DataLoader
    train_A = DataLoader(train_1, batch_size=parsed['batch_size'],drop_last=True)
    test_A = DataLoader(test_1, batch_size=parsed['batch_size'],drop_last=True)

    train_B = DataLoader(train_2, batch_size=parsed['batch_size'],drop_last=True)
    test_B = DataLoader(test_2, batch_size=parsed['batch_size'],drop_last=True)

    label_train = DataLoader(labels, batch_size=parsed['batch_size'],drop_last=True)
    label_val = DataLoader(labels_val, batch_size=parsed['batch_size'],drop_last=True)

    mulist_A = torch.tensor(mulist_A)
    '''
    Training
    '''
    model_A = partyA.build_model(parsed['input_b'],parsed['output_b'],parsed['hidden_b'],mulist_A,'relu',parsed['sigma'],parsed['lam'],parsed['dataset'])
    modelB_Bottom = partyB.build_bottom_model(parsed['input_b'],parsed['output_b'],parsed['hidden_b'],'relu',parsed['dataset'])
    model_Top = partyB.build_top_model(parsed['input_t'],parsed['output_t'],parsed['hidden_t'],parsed['dataset'])

    opt_A=torch.optim.SGD(params=model_A.parameters(),lr=parsed['lr_b'],weight_decay=parsed['wd_b'],momentum=parsed['mo_b'])
    opt_B=torch.optim.SGD(params=modelB_Bottom.parameters(),lr=parsed['lr_b'],weight_decay=parsed['wd_b'],momentum=parsed['mo_b'])
    opt_top = torch.optim.SGD(params=model_Top.parameters(),lr=parsed['lr_t'],weight_decay=parsed['wd_t'],momentum=parsed['mo_t'])

    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(parsed['epochs']):
        print(f"Epoch {epoch}")
        train_correct = 0
        train_total = 0
        model_A.train()
        modelB_Bottom.train()
        model_Top.train()
        for dataA, dataB, datal in zip(train_A, train_B, label_train):
            opt_A.zero_grad()
            opt_B.zero_grad()
            opt_top.zero_grad()

            a1,rega = model_A(dataA)
            a2 = modelB_Bottom(dataB)
            a1_detach = a1.clone().detach()
            a2_detach = a2.clone().detach()

            # random top-k
            W1 = construct_W(a1_detach,**kwargs_W)
            score1 = lap_score(a1_detach, W=W1)
            score1 = torch.tensor(score1)
            a1_detach, mask1 = getrandomk_lap(score1, a1_detach, floor(parsed['output_b']*parsed['rho']), parsed['p'])
            # add noise
            with torch.no_grad():
                l2 = torch.norm(a1_detach, p=2, dim=1)
                me1 = torch.mean(l2, dim=0).item()
                a1_clip = clip(a1_detach, me1)
                n1, sig = gaussion_noise(a1_clip, me1, parsed['dp_delta'], parsed['dp_epsilon'])
                a1_noise = a1_clip+n1

            a1_c = a1_noise.clone().detach().requires_grad_()
            a2_c = a2_detach.requires_grad_()
            d=[]
            d.append(a1_c)
            d.append(a2_c)
            agg_data, input_d = partyB.aggregation(d,'concatenate')

            y = model_Top(agg_data)

            criterion = nn.NLLLoss()
            loss = criterion(y, datal.long())

            train_total += y.shape[0]
            train_correct += y.max(1)[1].eq(datal).sum().item()
            loss.backward()

            g1 = a1_c.grad
            g2 = a2_c.grad

            g_detach = g1.clone().detach()
            g2_detach = g2.clone().detach()
            with torch.no_grad():
                g1_l2 = torch.norm(g_detach, p=2, dim=1)
                g_me = torch.mean(g1_l2, dim=0).item()
                g1_clip = clip(g_detach, g_me)
                gn, gsig = gaussion_noise(g1_clip, g_me, parsed['dp_delta'], parsed['dp_epsilon'])
                g1_noise = g1_clip+gn

            a1.backward(g1_noise)
            a2.backward(g2_detach)
            rega.backward()
            opt_top.step()
            opt_A.step()
            opt_B.step()

        model_A.eval()
        modelB_Bottom.eval()
        model_Top.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for dataA_v, dataB_v, label_v in zip(test_A, test_B, label_val):
                v1, _= model_A(dataA_v)
                v2 = modelB_Bottom(dataB_v)

                W_v1 = construct_W(v1, **kwargs_W)
                score_v1 = lap_score(v1, W=W_v1)
                score_v1 = torch.tensor(score_v1)
                v1, mask_v1 = getrandomk_lap(score_v1, v1, floor(parsed['output_b']*parsed['rho']), parsed['p'])

                mv = torch.norm(v1, p=2, dim=1)
                c_v1 = torch.mean(mv, dim=0).item()
                v1_clip = clip(v1, c_v1)
                vn, vsig = gaussion_noise(v1_clip, c_v1, parsed['dp_delta'], parsed['dp_epsilon'])
                v1_noise = v1_clip+vn

                v = torch.cat((v1_noise, v2), 1)
                vy = model_Top(v)

                correct += vy.max(1)[1].eq(label_v).sum().item()
                total += vy.shape[0]
        print(f"Train Accuracy: {100*train_correct/train_total:.3f}%")
        print(f"Val Accuracy: {100*correct/total:.3f}%")


