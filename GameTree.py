# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 15:28:45 2020

@author: escor
"""
import GameNode
from random import choice
from copy import deepcopy

class GameTree:
    
    def __init__(self, root=None, root_depth=1, expand_depth=4):
        
        if root == None:
            self.root = build_tree(None, expand_depth)
            
        else:
            self.root = build_tree(root, root.round, expand_depth)
            
        self.round_rooted = root_depth
        self.expanded_depth = expand_depth
        
def build_tree(node=None, expand_depth=1):
    
    if node == None:
        root = GameNode.GameNode(searchDepth=expand_depth)
        build_tree(root, expand_depth)
        return root
    
    elif node.isSubgameLeaf == True:
        root = deepcopy(node)
        root.subgameDepth = 1
        root.isSubgameLeaf = False
        root.isSubgameRoot = True
        root.get_valid_actions()
        build_tree(root, expand_depth)
        return root
        
    else:
        if node.isTerminal == True:
            node.legal_actions = None
            
        else:
            node.legal_actions = node.get_valid_actions()          
            for child in node.legal_actions:
                if child != None:
                    child_node = GameNode.GameNode(child, node, expand_depth)
                    build_tree(child_node,expand_depth)
                    node.children.append(child_node)
 

def sample_leaf(root):
    
    leaf_nodes = []
    
    def _sample_leaf(current, l_nodes):
        
        if current == None:
            return
        
        if current.isSubgameLeaf == True:
            leaf_nodes.append(current)
        
        try:
            for c in current.children:
                _sample_leaf(c, l_nodes)
        except:
            return
        
    _sample_leaf(root, leaf_nodes)
        
    leaf = choice(leaf_nodes)
    leaf.isTerminal = False
    leaf.isSubgameRoot = True    
    return leaf  
        
x = build_tree(expand_depth=1)
z = sample_leaf(x)
y = build_tree(z, expand_depth=1)
v = sample_leaf(y)
b = build_tree(v)
n = sample_leaf(b)
m = build_tree(n)