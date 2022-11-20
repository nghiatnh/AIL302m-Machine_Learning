from __future__ import annotations
import numpy as np
from typing import *
from collections import Counter
from ..utils.metrics import entropy, accuracy_score
import numbers


class DecisionNode():
    def __init__(self, *, ids: List[int] = None,
                 children: List[DecisionNode] = [],
                 entropy: float = 0.0,
                 depth: int = 0,
                 split_attribute: str | int = None,
                 values_order: List[str | int] = None,
                 rule: str = None,
                 label: str | int = None
                 ) -> None:
        '''
        Decision Node
        ------------

        The node for building a decision tree

        Parameter
        ----------

        ids: List[int], default = None
            List indexes of train data in this node

        children: List[DecisionNode], default = []
            Spitted children of this node

        entropy: float, default = 0
            Entropy of this node for further calculations, will fill later

        depth: int, default 0
            Distance from this node to root. The node with depth = 0 is root of the decision tree

        split_attribute: str | int, default = None
            Attribute using for split node

        value_order: List[str | int], default = None
            All possible values for splitting of child node

        label: str | int, default = None
            Label of node if it is a leaf node

        rule: {"==", "<=", ">"}, default = "=="
            The rule to compare check attribute and check value. E.g: "Overlook == sunny" or "X[2] <= 5"

            - If attribute type is categorical, rule always "=="
            - If attribute type is continuous, rule can be "<=" or ">"

        '''
        self.ids: List[int] = ids
        self.entropy: float = entropy
        self.depth: int = depth
        self.children: List[DecisionNode] = children

        self.rule: str = rule

        self.split_attribute: str | int = split_attribute
        self.values_order: List[str | int] = values_order
        self.label: str | int = label

    def set_properties(self, split_attribute: str | int, value_order: List[str | int]) -> None:
        '''
        Set properties of current node
        '''
        self.split_attribute = split_attribute
        self.values_order = value_order

    def set_label(self, label: str | int) -> None:
        '''
        Update labels of current node
        '''
        self.label = label

    def print(self, attributes_name: List[str | int] = None) -> None:
        '''
        Print tree nodes's information
        '''

        print("----" * self.depth)
        print("    " * self.depth + "|depth: {}".format(self.depth))
        print("    " * self.depth + "|ids: {}".format(self.ids))
        print("    " * self.depth + "|entropy: {}".format(self.entropy))
        att = (f"X[{self.split_attribute}]" if not attributes_name else attributes_name[self.split_attribute]
               ) if not self.split_attribute is None else None
        print("    " * self.depth + "|split attribute: {}".format(att))
        print("    " * self.depth + "|node rule: {}".format(self.rule))
        print("    " * self.depth + "|label: {}".format(self.label))
        for child in self.children:
            child.print(attributes_name)


class DecisionTreeClassifier():
    def __init__(self, *, max_depth: int = 10,
                 min_samples_split: int = 2,
                 min_gain: float = 1e-4, attributes_name: List[str | int] | None = None,
                 attribute_type: Literal["categorical",
                                         "continuous"] = 'categorical'
                 ) -> None:
        '''
        Decision Tree Classifier
        -----------

        Decision tree for classification problems using ID3 algorithm

        Parameter
        -----------

        max_depth: int, default = 10
            The maximum depth of tree

        min_samples_split: int, default = 2
            The minimum number of samples required to split an internal node

        min_gain: float, default 1e-4
            The minimum of information gain after each split step

        attributes_name: List[str | int], default = None
            The list of attributes name for printing:

            - If None, then print like X[i] for attributes name
            - If not None, then print like "Overlook == sunny"

        attribute_type: {"categorical", "continuous"}, default = "categorical"
            The type for split a node base on its attribute

            - If categorical, then use "==" for all values of variable X
            - If continuous, then use "<=" and ">" for only split value of variable x
        '''
        self.root: DecisionNode = None
        self.max_depth: int = max_depth
        self.min_samples_split: int = min_samples_split
        self.N: int = 0
        self.min_gain: float = min_gain
        self.attributes_name: List[str | int] = attributes_name
        self.attribute_type: str = attribute_type if attribute_type == 'categorical' or attribute_type == 'continuous' else 'categorical'

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        '''
        Fit and train model from given data.
        '''
        if self.attribute_type == 'categorical':
            if not np.all((isinstance(X[i,j], numbers.Number) for i in range(X.shape[0]) for j in range(X.shape[1]))):
                raise Exception("continuous only for numeric array")

        self.N = Y.size
        self.X: np.ndarray = X
        self.attributes: List[int] = [x for x in range(X.shape[1])]
        self.Y: np.ndarray = Y
        self.labels: np.ndarray = np.unique(Y)

        ids = [id for id in range(self.N)]
        self.root = DecisionNode(ids=ids, entropy=self.__entropy(ids), depth=0)
        queue = [self.root]

        while queue:
            node = queue.pop()
            if node.depth < self.max_depth or node.entropy < self.min_gain:
                node.children = self.__split(node)

                if not node.children:  # leaf node
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
            x = X[n, :]  # one point
            # start from root and recursively travel if not meet a leaf
            node = self.root
            while node.children:
                if self.attribute_type == "categorical":
                    node = node.children[node.values_order.index(
                        x[node.split_attribute])]
                elif x[node.split_attribute] > node.values_order[0]:
                    node = node.children[1]
                else:
                    node = node.children[0]
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
        if len(ids) == 0:
            return 0
        freq = np.array(list(Counter(self.Y[ids]).values()))
        return entropy(freq)

    def __set_label(self, node: DecisionNode) -> None:
        '''
        Find label for a node if it is a leaf. Simply chose by major voting.
        '''
        node.set_label(Counter(self.Y[node.ids]).most_common()[
                       0][0])  # most frequent label

    def __make_rule(self, check_attribute: str | int, values_order: List[str | int], i: int):
        rules = ["<=", ">"]
        check_att = (f"X[{check_attribute}]" if not self.attributes_name else self.attributes_name[check_attribute]
                    ) if not check_attribute is None else None

        if check_att is None:
            return None

        rule = "==" if self.attribute_type == "categorical" else rules[i]
        check_value = values_order[i] if self.attribute_type == 'categorical' else values_order[0]

        return f"{check_att} {rule} {check_value}"

    def __categorical(self, ids: List[int], sub_data: np.ndarray, att: str | int, values: List[str | int]) -> Tuple[List[int], List[int]]:
        '''
        Split for categorical option
        '''
        split_ids = []
        for val in values:
            sub_ids = np.array(ids)[np.where(
                sub_data[:, att] == val)[0]].tolist()
            split_ids.append(sub_ids)

        return (split_ids, values)

    def __continuous(self, ids: List[int], sub_data: np.ndarray, att: str | int, values: List[str | int]) -> Tuple[List[int], List[int]]:
        '''
        Split for continuous option
        '''
        split_ids = []
        min_entropy = np.Infinity
        best_value = None
        for val in values:
            sub_id1 = np.array(ids)[np.where(
                sub_data[:, att] <= val)[0]].tolist()
            sub_id2 = np.array(ids)[np.where(
                sub_data[:, att] > val)[0]].tolist()
            HxS = len(sub_id1) * self.__entropy(sub_id1) / len(ids)
            HxS += len(sub_id2) * self.__entropy(sub_id2) / len(ids)
            if HxS < min_entropy:
                min_entropy = HxS
                split_ids = [sub_id1, sub_id2]
                best_value = [val]

        return (split_ids, best_value)

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
            if len(values) == 1:
                continue  # entropy = 0
            split_ids, best_values = self.__categorical(
                ids, sub_data, att, values) if self.attribute_type == 'categorical' else self.__continuous(ids, sub_data, att, values)

            # don't split if a node has too small number of points
            if min(map(len, split_ids)) < self.min_samples_split:
                continue

            # information gain
            HxS = 0
            for split_id in split_ids:
                HxS += len(split_id) * self.__entropy(split_id) / len(ids)
            gain = node.entropy - HxS

            if gain < self.min_gain:
                continue  # stop if small gain

            if gain > best_gain:
                best_gain = gain
                best_splits = split_ids
                best_attribute = att
                values_order = best_values

        node.set_properties(best_attribute, values_order)

        child_nodes = [DecisionNode(ids=split_id,
                                    entropy=self.__entropy(split_id), depth=node.depth + 1,
                                    rule=self.__make_rule(node.split_attribute, values_order, i))
                       for i, split_id in enumerate(best_splits)]
        return child_nodes
