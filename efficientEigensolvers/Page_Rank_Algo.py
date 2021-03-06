import scraper

import Power_Iteration as pi
import numpy as np
import networkx as nx
import scipy

from matplotlib import pyplot as plt
import ../WebCrawler/scraper.scraper as scraper

def power_iteration_application_test():
    A = np.array([[1.5, 0.5], [0.5, 1.5]])
    print(A)
    norm_indicator = False
    dom_eigenvector, dom_eigenvalue = pi.PowerMethod(A, norm_indicator, 0.0001)
    print(dom_eigenvalue)
    print(dom_eigenvector)

    norm_indicator = True
    dom_eigenvector, dom_eigenvalue = pi.PowerMethod(A, norm_indicator, 0.0001)
    print(dom_eigenvalue)
    print(dom_eigenvector)



def page_rank_application_test():
    G = nx.DiGraph()
    G.add_edges_from([(1,2),(1,3),(1,4),(2,3),(2,4), (3, 1),(4,1), (4, 3)])
    nx.draw(G, with_labels=True)
    plt.savefig("page_rank_application_sample.png")
    plt.show()
    print("Limiting Distribution: ")
    print(PageRank(G, 0.15))

def stochastic_transition_matrix(G, weight, adaptive):

    Aj = nx.to_numpy_matrix(G).A
    N = len(G.nodes)
    v = np.empty()
    v.fill(1/N)

    P = np.zeros(shape=(N, N))
    dangling_notes = set()
    for j, node in enumerate(G.nodes()):
        out_deg = G.out_degree(node)
        if out_deg == 0:
            dangling_notes.add(out_deg)
        else:
            for i in range(N):
                P[i][j] = Aj[i][j] / out_deg


    if adaptive:
        return P
    else:
        #test dangling nodes
        d = np.empty()
        for i in range(N):
            if i in dangling_notes:
                d[i] = 1
        D = d.dot(v)




        S = np.ones(shape=(N, N))
        S = np.multiply(S, 1 / N)
    Aw = np.multiply(A, 1 - weight)
    Sw = np.multiply(S, weight)
    M = Aw + Sw
    return M



def multiplication_with_P(P, c, x):
    N = A.shape[0]
    v = np.empty(N)
    v.fill(1/N)
    y = np.multiply(P, c).dot(x)
    w = np.linalg.norm(x,1) - np.linalg.norm(y,1)
    y = y + np.multiply(v,w)
    return y


def PageRank(G, weight):

    M = stochastic_transition_matrix(G, weight)

    return pi.PowerMethod(M,True, 0.01)[0]



if __name__ == '__main__':
    print("###Power Iteration Test###")
    power_iteration_application_test()

    #print("###Page Rank Test###")
    #page_rank_application_test()



