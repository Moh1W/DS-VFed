import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    # Gini
    parser.add_argument('--epsilon', type=float, default=2.0, help="privacy budget of perturbing labels")
    parser.add_argument('--coefficient', type=float, default=0.1, help="inverse proportion coefficient")
    parser.add_argument('--sigma', type=float, default=1.0, help="standard deviation of the feature selection constant")
    parser.add_argument('--lam', type=float, default=0.1, help='coefficient of the regularization term')

    # federated arguments
    parser.add_argument('--epochs', type=int, default=40, help="rounds of training")
    parser.add_argument('--batch_size', type=int, default=128, help="batch size")

    parser.add_argument('--lr_b', type=float, default=0.05, help="learning rate")
    parser.add_argument('--wd_b', type=float, default=0.005, help="learning rate decay each round")
    parser.add_argument('--mo_b', type=float, default=0, help="SGD momentum")

    parser.add_argument('--lr_t', type=float, default=0.05, help="learning rate")
    parser.add_argument('--wd_t', type=float, default=0.005, help="learning rate decay each round")
    parser.add_argument('--mo_t', type=float, default=0, help="SGD momentum")
    parser.add_argument('--rho',type=float, default=0.9, help="1-rho")
    parser.add_argument('--p', type=float, default=0.9, help="Probability")
    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--input_b', type=int, default=392, help= 'input size of the bottom model')
    parser.add_argument('--hidden_b', type=int, nargs='+', default=None, help='hidden layers of the bottom model')
    parser.add_argument('--output_b', type=int, default=64, help='output size of the bottom model')

    parser.add_argument('--input_t', type=int, default=128, help= 'input size of the bottom model')
    parser.add_argument('--hidden_t', type=int, nargs='+', default=500, help='hidden layers of the bottom model')
    parser.add_argument('--output_t', type=int, default=10, help='output size of the bottom model')
    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")


    # DP
    parser.add_argument('--dp_epsilon', type=float, default=1.0,
                        help='differential privacy epsilon')
    parser.add_argument('--dp_delta', type=float, default=1e-4,
                        help='differential privacy delta')



    try: parsed = vars(parser.parse_args())
    except IOError as msg: parser.error(str(msg))
    return parsed
