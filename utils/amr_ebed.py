#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="7"


# In[118]:


import pandas as pd
import numpy as np
import pickle
import networkx as nx
import penman
from karateclub import Graph2Vec
import amrlib
import re,string


stog = amrlib.load_stog_model()


# In[13]:


from tqdm import tqdm


# In[14]:


train_df=pd.read_csv(r'./covid_dataset/Constraint_English_Train - Sheet1.csv')
dev_df=pd.read_csv(r'./covid_dataset/Constraint_English_Val - Sheet1.csv')
test_df=pd.read_csv(r'./covid_dataset/english_test_with_labels - Sheet1.csv')

# Concatenate the dataframes vertically
df = pd.concat([train_df, dev_df, test_df], axis=0)

# Reset the index of the combined dataframe
df.reset_index(drop=True, inplace=True)




#df['tweet']=df['tweet'].str.replace(r'http\S+', '').str.replace(r'@\S+','').str.replace(r'~\S+','').str.replace(r'#\S+','').str.replace(r'|','').str.replace(r',','').str.replace(r'-','').str.replace(r'\'','').str.replace(r'%','').str.replace(r'\d','').str.replace(r'\n','')
df['tweet']= df['tweet'].apply(lambda x: re.sub(r'[\n\r;@#~.!"()%]|http[s]?://\S+|', '', x))

df['no_of_sent']=0




# In[15]:


df


# In[18]:


# In[82]:


import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
df['no_of_sent']= df['tweet'].apply(lambda x: len(sent_tokenize(x)))
# for i in range(df.shape[0]):
#     df['no_of_sent'][i] = len(sent_tokenize(df['tweet'][i]))


# In[18]:


import networkx as nx

__all__ = ["convert_node_labels_to_integers", "relabel_nodes"]

def relabel_nodes(G, mapping, copy=True):
    m = {n: mapping(n) for n in G} if callable(mapping) else mapping

    if copy:
        return _relabel_copy(G, m)
    else:
        return _relabel_inplace(G, m)



def _relabel_inplace(G, mapping):
    if len(mapping.keys() & mapping.values()) > 0:
        # labels sets overlap
        # can we topological sort and still do the relabeling?
        D = nx.DiGraph(list(mapping.items()))
        D.remove_edges_from(nx.selfloop_edges(D))
        try:
            nodes = reversed(list(nx.topological_sort(D)))
        except nx.NetworkXUnfeasible as err:
            raise nx.NetworkXUnfeasible(
            ) from err
    else:
        # non-overlapping label sets, sort them in the order of G nodes
        nodes = [n for n in G if n in mapping]

    multigraph = G.is_multigraph()
    directed = G.is_directed()

    for old in nodes:
        # Test that old is in both mapping and G, otherwise ignore.
        try:
            new = mapping[old]
            G.add_node(new, **G.nodes[old])
        except KeyError:
            continue
        if new == old:
            continue
        if multigraph:
            new_edges = [
                (new, new if old == target else target, key, data)
                for (_, target, key, data) in G.edges(old, data=True, keys=True)
            ]
            if directed:
                new_edges += [
                    (new if old == source else source, new, key, data)
                    for (source, _, key, data) in G.in_edges(old, data=True, keys=True)
                ]
            # Ensure new edges won't overwrite existing ones
            seen = set()
            for i, (source, target, key, data) in enumerate(new_edges):
                if target in G[source] and key in G[source][target]:
                    new_key = 0 if not isinstance(key, (int, float)) else key
                    while new_key in G[source][target] or (target, new_key) in seen:
                        new_key += 1
                    new_edges[i] = (source, target, new_key, data)
                    seen.add((target, new_key))
        else:
            new_edges = [
                (new, new if old == target else target, data)
                for (_, target, data) in G.edges(old, data=True)
            ]
            if directed:
                new_edges += [
                    (new if old == source else source, new, data)
                    for (source, _, data) in G.in_edges(old, data=True)
                ]
        G.remove_node(old)
        G.add_edges_from(new_edges)
    return G


def _relabel_copy(G, mapping):
    H = G.__class__()
    H.add_nodes_from(mapping.get(n, n) for n in G)
    H._node.update((mapping.get(n, n), d.copy()) for n, d in G.nodes.items())
    if G.is_multigraph():
        new_edges = [
            (mapping.get(n1, n1), mapping.get(n2, n2), k, d.copy())
            for (n1, n2, k, d) in G.edges(keys=True, data=True)
        ]

        # check for conflicting edge-keys
        undirected = not G.is_directed()
        seen_edges = set()
        for i, (source, target, key, data) in enumerate(new_edges):
            while (source, target, key) in seen_edges:
                if not isinstance(key, (int, float)):
                    key = 0
                key += 1
            seen_edges.add((source, target, key))
            if undirected:
                seen_edges.add((target, source, key))
            new_edges[i] = (source, target, key, data)

        H.add_edges_from(new_edges)
    else:
        H.add_edges_from(
            (mapping.get(n1, n1), mapping.get(n2, n2), d.copy())
            for (n1, n2, d) in G.edges(data=True)
        )
    H.graph.update(G.graph)
    return H


def convert_node_labels_to_integers(
    G, first_label=0, ordering="default", label_attribute=None
):
    N = G.number_of_nodes() + first_label
    if ordering == "default":
        mapping = dict(zip(G.nodes(), range(first_label, N)))
    elif ordering == "sorted":
        nlist = sorted(G.nodes())
        mapping = dict(zip(nlist, range(first_label, N)))
    elif ordering == "increasing degree":
        dv_pairs = [(d, n) for (n, d) in G.degree()]
        dv_pairs.sort()  # in-place sort from lowest to highest degree
        mapping = dict(zip([n for d, n in dv_pairs], range(first_label, N)))
    elif ordering == "decreasing degree":
        dv_pairs = [(d, n) for (n, d) in G.degree()]
        dv_pairs.sort()  # in-place sort from lowest to highest degree
        dv_pairs.reverse()
        mapping = dict(zip([n for d, n in dv_pairs], range(first_label, N)))
    else:
        raise nx.NetworkXError(f"Unknown node ordering: {ordering}")
    H = relabel_nodes(G, mapping)
    # create node attribute with the old label
    if label_attribute is not None:
        nx.set_node_attributes(H, {v: k for k, v in mapping.items()}, label_attribute)
    return H


# In[89]:




# In[17]:


graphs_list=[]


# In[114]:
print(df.shape[0])

for i in tqdm(range(df.shape[0])):
    #print(i)
    sents = sent_tokenize(df['tweet'][i])
    if len(sents)>0:
        for j  in  range(len(sents)):
            #print(len(graphs))
            graphs=stog.parse_sents([sents[j]])
            gph = nx.Graph()
            #print(graphs[j])
            g = penman.decode(graphs[0])
            if len(g.edges())>0:
                for i in range(len(g.edges())):
                    gph.add_edges_from([(g.edges()[i][0],g.edges()[i][2])])
                gph=convert_node_labels_to_integers(gph, first_label=0, ordering='default', label_attribute=None)
                graphs_list.append(gph)
                #print(graphs_list)
            else:
                print('list is empty')
                graphs_list.append(gph)
                
    else:
        gph = nx.Graph()
        print('list is empty')
        graphs_list.append(gph)
        
   
with open('covid_fake_amr_graph_lst.pkl', 'wb') as f:
    pickle.dump(graphs_list, f)






