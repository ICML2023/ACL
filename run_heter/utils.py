import sys
sys.path.append("..")
from data import build_graph, utils
import seaborn as sb
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sklearn as sk
import networkx as nx
import torch.nn.functional as F



def mse_loss(predictions, targets):
    total_loss = torch.sqrt((predictions - targets).pow(2).mean(-1).mean(-1))
    return torch.sum(total_loss)

def contrastive_loss(predictions, output_emb, targets_pos, targets_neg, tau, lambda_loss):

    pos = torch.exp(torch.bmm(predictions, targets_pos.transpose(-1, -2)).squeeze()/tau)
    neg_score = torch.log(pos + torch.sum(torch.exp(torch.bmm(output_emb, targets_neg.transpose(-1, -2)).squeeze()/tau), dim=1).unsqueeze(-1))
    neg_score = torch.sum(neg_score, dim=1)
    pos_socre = torch.sum(torch.log(pos), dim=1)
    total_loss = torch.sum(lambda_loss * neg_score - pos_socre)
    total_loss = total_loss/targets_pos.shape[0]/targets_pos.shape[1]

    return total_loss

class NodeClassificationDataset(Dataset):
    def __init__(self, node_embeddings, labels):
        self.len = node_embeddings.shape[0]
        self.x_data = node_embeddings
        self.y_data = labels.long()

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def cluster_graph(role_id, node_embeddings):
    colors = role_id
    nb_clust = len(np.unique(role_id))
    pca = PCA(n_components=5)
    trans_data = pca.fit_transform(StandardScaler().fit_transform(node_embeddings.cpu().detach()))
    km = KMeans(n_clusters=nb_clust)
    km.fit(trans_data)
    labels_pred = km.labels_

    ######## Params for plotting
    cmapx = plt.get_cmap('rainbow')
    x = np.linspace(0, 1, nb_clust + 1)
    col = [cmapx(xx) for xx in x]
    markers = {0: '*', 1: '.', 2: ',', 3: 'o', 4: 'v', 5: '^', 6: '<', 7: '>', 8: 3, 9: 'd', 10: '+', 11: 'x',
               12: 'D', 13: '|', 14: '_', 15: 4, 16: 0, 17: 1, 18: 2, 19: 6, 20: 7}

    for c in np.unique(role_id):
        indc = [i for i, x in enumerate(role_id) if x == c]
        plt.scatter(trans_data[indc, 0], trans_data[indc, 1],
                    c=np.array(col)[list(np.array(labels_pred)[indc])],
                    marker=markers[c % len(markers)], s=300)

    labels = role_id
    for label, c, x, y in zip(labels, labels_pred, trans_data[:, 0], trans_data[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    # plt.show()
    return labels_pred, colors, trans_data, nb_clust


def unsupervised_evaluate(colors, labels_pred, trans_data, nb_clust):
    ami = sk.metrics.adjusted_mutual_info_score(colors, labels_pred)
    sil = sk.metrics.silhouette_score(trans_data, labels_pred, metric='euclidean')
    ch = sk.metrics.calinski_harabasz_score(trans_data, labels_pred)
    hom = sk.metrics.homogeneity_score(colors, labels_pred)
    comp = sk.metrics.completeness_score(colors, labels_pred)
    #print('Homogeneity \t Completeness \t AMI \t nb clusters \t CH \t  Silhouette \n')
    #print(str(hom) + '\t' + str(comp) + '\t' + str(ami) + '\t' + str(nb_clust) + '\t' + str(ch) + '\t' + str(sil))
    return hom, comp, ami, nb_clust, ch, sil


def draw_pca(role_id, node_embeddings, coloring):
    pca = PCA(n_components=2)
    node_embedded = StandardScaler().fit_transform(node_embeddings.cpu().detach())
    principalComponents = pca.fit_transform(node_embedded)
    principalDf = pd.DataFrame(data=principalComponents,
                               columns=['principal component 1', 'principal component 2'])
    principalDf['target'] = role_id
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 PCA Components', fontsize=20)
    targets = np.unique(role_id)
    for target in zip(targets):
        color = coloring[target[0]]
        indicesToKeep = principalDf['target'] == target
        ax.scatter(principalDf.loc[indicesToKeep, 'principal component 1'],
                   principalDf.loc[indicesToKeep, 'principal component 2'],
                   s=50,
                   c=color)
    ax.legend(targets)
    ax.grid()
    plt.show()


def graph_generator(width_basis=15, basis_type = "cycle", n_shapes = 5, shape_list=[[["house"]]], identifier = 'AA', add_edges = 0):
    ################################### EXAMPLE TO BUILD A SIMPLE REGULAR STRUCTURE ##########
    ## REGULAR STRUCTURE: the most simple structure:  basis + n small patterns of a single type
    ### 1. Choose the basis (cycle, torus or chain)
    ### 2. Add the shapes
    list_shapes = []
    for shape in shape_list:
        list_shapes += shape * n_shapes
    print(list_shapes)

    ### 3. Give a name to the graph
    name_graph = 'houses' + identifier
    sb.set_style('white')

    ### 4. Pass all these parameters to the Graph Structure
    G, communities, plugins, role_id = build_graph.build_structure(width_basis, basis_type, list_shapes, start=0,
                                                                   add_random_edges=add_edges,
                                                                   plot=True, savefig=False)
    return G, role_id


def average(lst):
    return sum(lst) / len(lst)


def write_graph2edgelist(G, role_id, filename):
    nx.write_edgelist(G, "{}.edgelist".format(filename), data=False)
    with open("{}.roleid".format(filename), "w") as f:
        for id in role_id:
            f.write(str(id) + "\n")

def set_pca( pca, embeddings):
    node_embedded = StandardScaler().fit_transform(embeddings)
    pca.fit(node_embedded)
    return pca
