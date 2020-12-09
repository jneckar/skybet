# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 21:50:46 2020

@author: escor
"""
import numpy as np
import copy
import HandManager as hm
        
class PBS:
    
    def __init__(self, root, p0_prob, p1_prob):
        
        self.active_player = root.current_player
        self.stack_pot_ratio = root.chips_remaining / (root.chips_bet * 2)
        
        

        
    def normalize_reach_prob(self):
        max_p0 = np.amax(self.p0_reach)
        max_p1 = np.amax(self.p1_reach)
        
        if max_p0 < 0.95:
            norm_factor = 1 / max_p0
            self.p0_reach *= norm_factor
            
        if max_p1 < 0.95:
            norm_factor = 1 / max_p1
            self.p1_reach *= norm_factor