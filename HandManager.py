# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 16:54:47 2020

@author: escor
"""

import numpy as np

class HandManager:
    
    CARD_RANKS = ['A','K','Q','J','T','9','8','7','6','5','4','3','2']
    CARD_SUITS = ['s','h','d','c']
    DECK = []
    
    for r in CARD_RANKS:
        for s in CARD_SUITS:
            DECK.append(r+s)
    
    hands_2s, hands_1s, hands_os = [np.array([], dtype = np.int8, ndmin=3)] * 3

    idx_to_str = {}
    str_to_idx = {}
    idx_to_cards = {}
    hand_iso_idx = {}
    card_removal = {}
    for i in range(52):
        card_removal[i] = np.array([],dtype=np.uint16)
    hand_removal = {}
    
    for i in range(13):
        for j in range(i,13):
            hand_iso = CARD_RANKS[i]+CARD_RANKS[j]
            if hand_iso[0] == hand_iso[1]:
                hand_iso_idx[hand_iso] = []
            else:
                hand_iso_idx[hand_iso+'s'] = []
                hand_iso_idx[hand_iso+'o'] = []
                
    hand_counter = 0
    for i in range(51):
        i_str = CARD_RANKS[i // 4] + CARD_SUITS[i % 4]
        for j in range(i+1, 52):
            j_str = CARD_RANKS[j // 4] + CARD_SUITS[j % 4]
            hand_str = i_str + j_str
            hand_iso = hand_str[0]+hand_str[2]
            if hand_iso[0] == hand_iso[1]:
                hand_iso_idx[hand_iso].append(hand_counter)
            elif hand_str[1] == hand_str[3]:
                hand_iso_idx[hand_iso+'s'].append(hand_counter)
            else:
                hand_iso_idx[hand_iso+'o'].append(hand_counter)
            hand_idx = hand_counter
            card_removal[i] = np.append(card_removal[i],hand_idx)
            card_removal[j] = np.append(card_removal[j],hand_idx)
            hand_counter += 1
            idx_to_cards[hand_idx] = [i,j]
            idx_to_str[hand_idx] = hand_str
            str_to_idx[hand_str] = hand_idx
            
    hand_counter = 0
    for i in range(51):
        for j in range(i+1,52):
            hand_removal[hand_counter] = np.array(list(set(list(card_removal[i])+list(card_removal[j]))),dtype=np.uint16)
            hand_counter += 1
            
    hand_mask = np.zeros((1326,1326), dtype=bool)
    for i in range(1326):
        hand_mask[i][hand_removal[i]] = True
                
    def apply_removal(self, reach_probs, to_remove):
        assert len(reach_probs) == 1326
        for card in to_remove:
            reach_probs[self.card_removal[card]] = 0
            
    def mask_blocked_hands(self, hand_array, mask=hand_mask):
        hand_array.mask = np.ma.mask_or(hand_array.mask, mask)
            
