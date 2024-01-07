import numpy as np
from model import *
from tools import *

if __name__ == '__main__':
    # memory capacity
    rng = np.random.RandomState(42)

    N = 1
    data_initial = (rng.rand(20000, N))
    tuo = 200
    MC = np.zeros((tuo))
    i = tuo + 1
    train_data = data_initial[i:(16000 + i), :]
    test_data = data_initial[(16000 + i):, :]

    N = data_initial.shape[1]
    Dr = 60
    density = 0.1
    rho = 1
    delta = 0.01
    b = 0
    transient = 1000
    # input-to-reservoir matrix
    W_in = rng.rand(Dr, N) * 2 - 1

    initial_matrix = np.dot(W_in, train_data[transient:, :].T)
    result_max = np.max(initial_matrix)
    result_min = np.min(initial_matrix)
    intial_expand = 0.5 * (result_max + result_min)
    initial_radius = 0.5 * (result_max - result_min)
    b = 0 - delta / initial_radius * intial_expand
    delta = delta / initial_radius


    network_name = ["ER", 'DCG', 'DAG']
    index_R = 0
    Network_weight = rng.rand(Dr, Dr)
    MC_configure = None
    R_network_0 = (
        Network_initial(network_name[index_R], network_size=Dr, density=density, Depth=0, MC_configure=MC_configure)).T
    R_network_0 = np.triu(np.multiply(R_network_0, Network_weight)) + np.triu(
        np.multiply(R_network_0, Network_weight)).T
    R_network = R_network_0 / np.max(np.abs(np.linalg.eigvals(R_network_0)))

    # top 10 nodes are the source nodes
    # W_in[MC_configure['number'][0]:, :] = 0

    # souce_node_index = np.ones(Dr)
    # souce_node_index[MC_configure['number'][0]:] = 0

    rg = nx.from_numpy_array(((R_network > 0) * 1).T, create_using=nx.DiGraph())
    # print(pd.Series(node_cluster1(rg, souce_node_index)).value_counts())

    RC = reservoir_computing(
        N=N,
        Dr=Dr,
        rho=rho,
        delta=delta,
        b=b,
        transient=transient,
        R_network=R_network,
        W_in=W_in)

    MC_C = np.zeros((tuo))
    for i in range(tuo, 0, -1):
        # print(i)
        Expect_output = data_initial[i:(16000+(i)), :]
        print(Expect_output.shape)
        pred_train = RC.Training_phase(train_data, Expect_output, index_method=4)

        # print(pred_train.shape, Expect_output.shape)

        MC_C[tuo - i] = memory_capacity(pred_train, Expect_output, length=100)
        # print(MC_C[tuo - i])
    plt.plot(MC_C)
    print(np.sum(MC_C[MC_C > 0.1]))
