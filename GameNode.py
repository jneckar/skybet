# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 18:26:01 2020

@author: escor
"""

from copy import deepcopy

BIG_BLIND = 100
START_STACK = 5000

class GameNode:
    
    def __init__(self, action=None, parent=None, searchDepth=1):
        
        if parent == None: #start new hand
            self.hist = ''
            self.chips_bet = 0
            self.chips_remaining = START_STACK
            self.bet = [0,BIG_BLIND]
            self.round = 1
            self.parent = None
            self.current_player = 0
            self.searchDepth = searchDepth
            self.subgameDepth = 1
            
        else:
            self.hist = str(parent.hist) + str(action)[0]
            if action[0] == 'b':
                self.hist += action[1:]
            self.chips_bet = deepcopy(parent.chips_bet)
            self.chips_remaining = deepcopy(parent.chips_remaining)
            self.bet = deepcopy(parent.bet)
            self.round = deepcopy(parent.round)
            self.parent = parent
            self.current_player = deepcopy(parent.current_player)
            self.searchDepth = searchDepth
            self.subgameDepth = deepcopy(parent.subgameDepth)
            
        print(self.hist)
        
        self.children = []
        self.isSubgameLeaf = False
        self.isSubgameRoot = False
        self.isTerminal = False
        
        self._action_handler(action)
        
                    
    def _get_valid_betsizes(node):
        
        if node.hist == '':
            return (BIG_BLIND, int(BIG_BLIND * 1.5), BIG_BLIND * 2)
        
        min_raise = None
        pot_raise = None
        max_raise = None
        facing_bet = node.bet[(node.current_player + 1) % 2]
        
        if node.current_player not in [0,1] or node.chips_remaining - facing_bet == 0:
            return (min_raise, pot_raise, max_raise)
        

        min_raise = min(node.chips_remaining - facing_bet ,max(BIG_BLIND, 2 * facing_bet + node.chips_bet))
        max_raise = node.chips_remaining - facing_bet
        pot_raise = min(max_raise, 2 * (facing_bet + node.chips_bet) + facing_bet)
        if node.round == 1 and node.current_player == 0 and node.bet[1] == BIG_BLIND:
            min_raise = BIG_BLIND * 2
            
        if pot_raise == min_raise or pot_raise == max_raise:
            pot_raise = None
        if max_raise == min_raise:
            max_raise = None
        
        return (min_raise, pot_raise, max_raise)
            
    
    def get_valid_actions(node):
        
        if node.current_player == 2:
            return ['_deal']
        
        if node.isTerminal == True:
            return [None]
        
        valid = [None] * 5
        valid[0] = 'call' #synonymous to check
        if node.bet[node.current_player] < node.bet[(node.current_player + 1) % 2]:
            valid[1] = 'fold'
        valid_raises = node._get_valid_betsizes()
        for i in range(3):
            if valid_raises[i] != None:
                valid[i+2] = 'b' + str(valid_raises[i])
        
        return valid
    
    def _action_handler(node, action):
        
        if action == '_deal':
            if node.searchDepth == node.subgameDepth:
                node.isSubgameLeaf = True
                node.isTerminal = True
            node.round += 1
            node.subgameDepth += 1
            node.current_player = 1
            
        elif action == 'call':
                
            if node.round == 1:
                if node.hist == 'c':
                    node.chips_bet = BIG_BLIND
                    node.chips_remaining -= BIG_BLIND
                    node.bet = [0,0]
                    node.current_player = 1
                else:
                    node.chips_bet += max(node.bet)
                    node.chips_remaining -= max(node.bet)
                    node.bet = [0,0]
                    if node.chips_remaining != 0:
                        node.current_player = 2
                    else:
                        node.isTerminal = True
                    
            elif node.current_player == 1 and node.bet[0] == 0:
                node.current_player = 0
                
            elif node.chips_remaining - max(node.bet) == 0:
                node.chips_bet = START_STACK
                node.chips_remaining = 0
                node.bet = [0,0]
                node.isTerminal = True
            
            else:
                node.chips_bet += max(node.bet)
                node.chips_remaining -= max(node.bet)
                node.bet = [0,0]
                assert node.chips_remaining >= 0
                
                if node.round == 4:
                    node.isTerminal = True
                    node.isSubgameLeaf = True
                else:
                    node.current_player = 2
                    
                    
        elif action == 'fold':
            node.chips_bet += min(node.bet)
            if node.round == 1 and node.current_player == 0:
                node.chips_bet == BIG_BLIND // 2
            node.isTerminal = True
            
        elif action is not None and action[0] == 'b':
            node.bet[node.current_player] = int(action[1:])
            if node.bet[(node.current_player + 1) % 2] != 0:
                node.chips_bet += node.bet[(node.current_player + 1) % 2]
                node.chips_remaining -= node.bet[(node.current_player + 1) % 2]
                node.bet[(node.current_player + 1) % 2] = 0
            node.current_player = (node.current_player + 1) % 2
            
        else:
            pass      
        
        #assert node.chips_remaining - max(node.bet) >= 0
        
        
#root = GameNode(searchDepth=2)
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                