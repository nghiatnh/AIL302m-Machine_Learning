from __future__ import annotations
import numpy as np
from typing import *
from collections import Counter
from ..utils.metrics import entropy, accuracy_score

class DecisionNode():
    def __init__(self, ids: List[int] = None, children: List[DecisionNode] = [], entropy: float = 0.0, depth: int = 0) -> None:
        self.ids: List[int] = ids           # index of data in this node
        self.entropy: float = entropy   # entropy, will fill later
        self.depth: int = depth       # distance to root node
        self.split_attribute: str | int = None # which attribute is chosen, it non-leaf
        self.children: List[DecisionNode] = children # list of its child nodes
        self.values_order: List[str | int] = None       # order of values of split_attribute in children
        self.label: str | int = None       # label of node if it is a leaf

    def set_properties(self, split_attribute : str | int, value_order : List[str | int]) -> None:
        self.split_attribute = split_attribute
        self.values_order = value_order

    def set_label(self, label: str | int) -> None:
        self.label = label

    def print(self, attributes_name: List[str | int] = None) -> None:
        '''
        Print tree nodes's information
        '''
        print("----" * self.depth)
        print("    " * self.depth + "|ids: {}".format(self.ids))
        print("    " * self.depth + "|entropy: {}".format(self.entropy))
        print("    " * self.depth + "|label: {}".format(self.label))
        print("    " * self.depth + "|attribute: {}".format((self.split_attribute if not attributes_name else attributes_name[self.split_attribute]) if not self.split_attribute is None else None))
        for child in self.children:
            child.print(attributes_name)

class DecisionTreeClassifier():
    def __init__(self, max_depth: int = 10, min_samples_split: int = 2, min_gain: float = 1e-4, attributes_name: List[str | int] | None = None):
        self.root: DecisionNode = None
        self.max_depth: int = max_depth 
        self.min_samples_split: int = min_samples_split 
        self.N: int = 0
        self.min_gain: float = min_gain
        self.attributes_name: List[str | int] = attributes_name
    
    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        '''
        Fit and train model from given data.
        '''
        self.N = Y.size
        self.X: np.ndarray = X
        self.attributes: List[int] = [x for x in range(X.shape[1])]
        self.Y: np.ndarray = Y
        self.labels: np.ndarray = np.unique(Y)
        
        ids = [id for id in range(self.N)]
        self.root = DecisionNode(ids = ids, entropy = self.__entropy(ids), depth = 0)
        queue = [self.root]

        while queue:
            node = queue.pop()
            if node.depth < self.max_depth or node.entropy < self.min_gain:
                node.children = self.__split(node)
                
                if not node.children: #leaf node
                    self.__set_label(node)
                else:
                    queue += node.children
            else:
                self.__set_label(node)

    def predict(self, X: np.ndarray) -> np.ndarray:
        '''
        Return predict output of given data 
        '''
        N = X.shape[0]
        labels = [None] * N
        for n in range(N):
            x = X[n, :] # one point 
            # start from root and recursively travel if not meet a leaf 
            node = self.root
            while node.children:
                node = node.children[node.values_order.index(x[node.split_attribute])]
            labels[n] = node.label
            
        return np.array(labels)

    def score(self, X: np.ndarray, Y: np.ndarray) -> float:
        '''
        Return true classification score
        '''
        return accuracy_score(self.predict(X), Y)

    def print_tree(self):
        '''
        Print tree nodes's information
        '''
        print("----DECISION TREE----")
        self.root.print(self.attributes_name)
                
    def __entropy(self, ids: List[int]) -> float:
        '''
        Calculate entropy of a node with index ids
        '''
        if len(ids) == 0: return 0
        freq = np.array(list(Counter(self.Y[ids]).values()))
        return entropy(freq)

    def __set_label(self, node: DecisionNode) -> None:
        '''
        Find label for a node if it is a leaf. Simply chose by major voting.
        '''
        node.set_label(Counter(self.Y[node.ids]).most_common()[0][0]) # most frequent label
    
    def __split(self, node: DecisionNode) -> List[DecisionNode]:
        '''
        Split current node to children
        '''
        ids = node.ids
        best_gain = 0
        best_splits: List[int] = []
        best_attribute: str | int = None
        values_order: List[str | int] = None
        sub_data = self.X[ids, :]
        for i, att in enumerate(self.attributes):
            values: List[str | int] = np.unique(self.X[ids, i]).tolist()
            if len(values) == 1: continue # entropy = 0
            split_ids = []
            for val in values:
                sub_ids = np.array(ids)[np.where(sub_data[:,att] == val)[0]].tolist()
                split_ids.append(sub_ids)
            # don't split if a node has too small number of points
            if min(map(len, split_ids)) < self.min_samples_split: continue
            
            # information gain
            HxS= 0
            for split_id in split_ids:
                HxS += len(split_id) * self.__entropy(split_id) / len(ids)
            gain = node.entropy - HxS 

            if gain < self.min_gain: continue # stop if small gain 
            
            if gain > best_gain:
                best_gain = gain 
                best_splits = split_ids
                best_attribute = att
                values_order = values
        
        node.set_properties(best_attribute, values_order)

        child_nodes = [DecisionNode(ids = split_id,
                     entropy = self.__entropy(split_id), depth = node.depth + 1) for split_id in best_splits]
        return child_nodes
