# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 23:02:55 2020

@author: escor
"""

import numpy as np
import numpy.ma as ma
from itertools import combinations, combinations_with_replacement
#from collections import defaultdict
import random
from scipy.sparse import csr_matrix
import pickle
import os
import HandManager

class LookupTable:
    
    def __init__(self, path='./lookup/'):
        
        self.path = path
        self.eval_path = path+'eval_table.pkl'
        self.iso_path = path+'isomorphs.pkl'
        #self.util_path = path+'showdown.pkl'
        self.hm = HandManager.HandManager()
        
        self.k_5, self.k_3, self.k_2_12, self.k_2_13 = self._get_rank_kickers()
        
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        
        if not os.path.exists(self.eval_path):
            eval_outputs = self._build_eval_table()
            print('Lookup tables built successfully. Saving to ', self.eval_path)
            eval_file = open(self.eval_path, 'wb')
            pickle.dump(eval_outputs, eval_file)
            eval_file.close()
            
        else:
            eval_file = open(self.eval_path, mode='rb')
            eval_outputs = pickle.load(eval_file)
            eval_file.close()
            
        self.hands_os = eval_outputs[0]
        self.hands_2s = eval_outputs[1]
        self.hands_1s = eval_outputs[2]
        self.lookup_os = eval_outputs[3][91]
        self.lookup_3s = eval_outputs[3][169]
        self.lookup_45s = eval_outputs[3][338]
        self.lookup = eval_outputs[3]
        self.lookup.update({0: self.lookup_os, 3: self.lookup_3s, 
                            4: self.lookup_45s, 5: self.lookup_45s})
            
        if not os.path.exists(self.iso_path):
            iso_outputs = self._build_isomorph_indexes()
            iso_file = open(self.iso_path, 'wb')
            pickle.dump(iso_outputs, iso_file)
            iso_file.close()
            
        else:
            iso_file = open(self.iso_path, mode = 'rb')
            iso_outputs = pickle.load(iso_file)
            iso_file.close()

        self.isomorphs_os_idx = iso_outputs[0]
        self.isomorphs_2s_idx = iso_outputs[1]
        self.isomorphs_1s_idx = iso_outputs[2]
        self.iso_idx = {0: self.isomorphs_os_idx, 3: self.isomorphs_2s_idx, 
                        4: self.isomorphs_1s_idx, 5: self.isomorphs_1s_idx,
                        91: self.isomorphs_os_idx, 169: self.isomorphs_2s_idx,
                          338: self.isomorphs_1s_idx}
        
        self.translate_iso_os = [iso_outputs[3]]
        self.translate_iso_2s = iso_outputs[4]
        self.translate_iso_1s = iso_outputs[5]
        self.translate = {0: self.translate_iso_os, 3: self.translate_iso_2s,
                          4: self.translate_iso_1s, 5: self.translate_iso_1s, 
                          91: self.translate_iso_os, 169: self.translate_iso_2s,
                          338: self.translate_iso_1s}

        '''if not os.path.exists(self.util_path):
            util_outputs = self._build_isomorph_indexes()
            iso_file = open(self.iso_path, 'wb')
            pickle.dump(iso_outputs, iso_file)
            iso_file.close()
            
        else:
            iso_file = open(self.iso_path, mode = 'rb')
            iso_outputs = pickle.load(iso_file)
            iso_file.close()   '''     

        
        
    def _build_eval_table(self):
        
        print('Lookup tables not found, rebuilding. This will take a while.')
        print('Enumerating hands and boards. . .')
        eval_os = {} 
        eval_3s = {} 
        eval_45s = {} 
        eval_table = {0: eval_os, 3: eval_3s, 4: eval_45s, 5: eval_45s, 
                      91: eval_os, 169: eval_3s, 338: eval_45s}

        allb, allh, hands_2s, hands_1s, hands_os = [np.array([], dtype = np.int8, ndmin=3)] * 5
        
        
        for comb in combinations(range(26),2):
            hand = np.zeros(26,dtype=np.int8)
            hand[list(comb)] = 1
            hand = np.reshape(hand, (2,13))
            hand[1] += hand[0]
            allh = np.append(allh,hand)
            if np.sum(hand[0]) == 2:
                hands_2s = np.append(hands_2s, hand)
            elif np.sum(hand[0] == 1):
                hands_1s = np.append(hands_1s, hand)
            else:
                hands_os = np.append(hands_os, hand)
                
        for pairs in range(13):
            pair = np.zeros((2,13),dtype=np.int8)
            pair[1][pairs] = 2
            allh = np.append(allh, pair)
            hands_os = np.append(hands_os, pair)
        
            
        for comb in combinations(range(13), 5):
            board = np.zeros((2,13), dtype=np.int8)
            board[:,list(comb)] = 1
            allb = np.append(allb, board)
            
        for s_comb in combinations(range(13), 4):
            suited = list(s_comb)
            for o_comb in range(13):
               board = np.zeros((2,13), dtype=np.int8)
               board[:,suited] = 1
               board[1][o_comb] += 1
               allb = np.append(allb, board)
               
        for s_comb in combinations(range(13), 3):
            suited = list(s_comb)
            for o_comb in combinations_with_replacement(range(13), 2):
                board = np.zeros((2,13), dtype=np.int8)
                board[:,suited] = 1
                for c in o_comb:
                    board[1][c] += 1
                allb = np.append(allb, board)
        
        for o_comb in combinations_with_replacement(range(13), 5):
            if len(set(o_comb)) != 1:
                board = np.zeros((2,13), dtype=np.int8)
                for c in o_comb:
                    board[1][c] += 1    
                allb = np.append(allb, board)          
        
        allh = np.reshape(allh, (-1,2,13)) 
        allb = np.reshape(allb, (-1,2,13))           
        hands_1s = np.reshape(hands_1s, (-1,2,13))
        hands_2s = np.reshape(hands_2s, (-1,2,13))
        hands_os = np.reshape(hands_os, (-1,2,13))
        
        done = 0
        num_os = len(hands_os)
        num_1s = len(hands_1s)
        num_2s = len(hands_2s)
        
        print('done.\nStarting hand evaluations. . .')
        
        for b in allb:
            
            if np.sum(b[0]) == 0:
                evals = np.empty(num_os, dtype=np.int16)
                for i in range(num_os):
                    evals[i] = self.evaluate_hand(b + hands_os[i])
                eval_os[b.tobytes()] = self._normalize_rank(evals)
                
            elif np.sum(b[0]) == 3:
                evals = np.empty(num_2s + num_os, dtype=np.int16)
                for i in range(num_os):
                    evals[i] = self.evaluate_hand(b + hands_os[i])                
                for i in range(num_os, num_os + num_2s):
                    evals[i] = self.evaluate_hand(b + hands_2s[i-num_os])
                eval_3s[b.tobytes()] = self._normalize_rank(evals)
                
            else:
                evals = np.empty(num_os + num_1s + num_2s, dtype=np.int16)
                for i in range(num_os):
                    evals[i] = self.evaluate_hand(b + hands_os[i])                
                for i in range(num_os, num_os + num_2s):
                    evals[i] = self.evaluate_hand(b + hands_2s[i-num_os])                
                for i in range(num_os + num_2s, num_os + num_2s + num_1s):
                    evals[i] = self.evaluate_hand(b + hands_1s[i-num_os-num_2s])
                eval_45s[b.tobytes()] = self._normalize_rank(evals)

            done += 1
            if done % 1000 == 0:
                print('complete: ', done, 'of 42783 boards')
        
        return [hands_os, hands_2s, hands_1s, eval_table]
    
    def evaluate_hand(self, hand):
        assert np.shape(hand) == (2,13)
        assert np.sum(hand[1]) in [5,6,7]
        
        if 2 in hand[0] or 5 in hand[1] or 6 in hand[1] or 7 in hand[1]:
            return -1
        
        if np.sum(hand[0]) < 5:
            has_flush = False
            
        else:
            for i in range(9):
                if 0 not in hand[0][i:i+5]:
                    return i+1
            if 0 not in hand[0][9:] and hand[0][0] == 1:
                return 10
            has_flush = True
        
        quads_rank = np.where(hand[1] == 4)[0]
        trips_rank = np.where(hand[1] == 3)[0]
        pair_rank = np.where(hand[1] == 2)[0]
        single_rank = np.where(hand[1] == 1)[0]
        
        if 4 in hand[1]:
            kicker_rank = min(np.concatenate((trips_rank, pair_rank, single_rank)))
            if kicker_rank > quads_rank[0]:
                kicker_rank -= 1
            return 11 + quads_rank[0] * 12 + kicker_rank
        
        if len(trips_rank) == 2:
            return 167 + trips_rank[0] * 12 + trips_rank[1]
        if (3 in hand[1] and 2 in hand[1]):
            p_rank = pair_rank[0]
            if p_rank > trips_rank[0]:
                p_rank -= 1
            return 167 + trips_rank[0] * 12 + pair_rank[0]
        
        if has_flush == True:
            return 323 + self.k_5[np.where(hand[0] == 1)[0][:5].astype(np.int8).tobytes()]

        for i in range(9):
            if 0 not in hand[1][i:i+5]:
                return 1600 + i
            if 0 not in hand[1][9:] and hand[1][0] != 0:
                return 1609
            
        if 3 in hand[1]:
            kickers = single_rank[:2]
            if kickers[0] > trips_rank[0]:
                kickers[0] -= 1
            if kickers[1] > trips_rank[0]:
                kickers[1] -= 1
            return 1610 + trips_rank[0] * 66 + self.k_2_12[kickers.astype(np.int8).tobytes()]
        
        if len(pair_rank) >= 2:
            if len(pair_rank) == 3:
                kicker_rank = min(pair_rank[2], single_rank[0])
            else:
                kicker_rank = single_rank[0]
            if kicker_rank > pair_rank[1]:
                kicker_rank -= 1
            if kicker_rank > pair_rank[0]:
                kicker_rank -= 1
            return 2468 + self.k_2_13[pair_rank[:2].astype(np.int8).tobytes()] * 11 + kicker_rank
        
        if len(pair_rank) == 1:
            kickers = single_rank[:3]
            for i in range(3):
                if kickers[i] > pair_rank[0]:
                    kickers[i] -= 1
            return 3326 + pair_rank[0] * 220 + self.k_3[kickers.astype(np.int8).tobytes()]
        
        kickers = single_rank[:5]
        return 6186 + self.k_5[kickers.astype(np.int8).tobytes()]
    
    def _build_isomorph_indexes(self):
        #maps all 1326 2 card hands to concat(hands_os, hands_3s, hands_45s)
        
        idx_full_to_os_iso = np.empty((1326), dtype=np.int16) #hand isomorph indexes for board with no flush possible
        idx_full_to_2s_iso = np.empty((4,1326), dtype=np.int16) #' ' 3 suited cards
        idx_full_to_1s_iso = np.empty((4,1326), dtype=np.int16) #' ' 4 or 5 suited cards
        
        idx_full_to_os_iso.fill(-1)
        idx_full_to_2s_iso.fill(-1)
        idx_full_to_1s_iso.fill(-1)
        
        idx = 0
        hands_iso = np.concatenate((self.hands_os, self.hands_2s, self.hands_1s))
        
        deck = ['As', 'Ah', 'Ad', 'Ac', 'Ks', 'Kh', 'Kd', 'Kc', 'Qs', 'Qh', 
                'Qd', 'Qc', 'Js', 'Jh', 'Jd', 'Jc', 'Ts', 'Th', 'Td', 'Tc', 
                '9s', '9h', '9d', '9c', '8s', '8h', '8d', '8c', '7s', '7h', 
                '7d', '7c', '6s', '6h', '6d', '6c', '5s', '5h', '5d', '5c', 
                '4s', '4h', '4d', '4c', '3s', '3h', '3d', '3c', '2s', '2h', 
                '2d', '2c']
        
        for comb in combinations(deck, 2):

                            
            i = deck.index(list(comb)[0])
            j = deck.index(list(comb)[1])
            
            conv_os = self._convert_hand([i,j])
            conv_s = [self._convert_hand([i,j], board_suit = k) for k in range(4)]
        
            for k in range(len(self.hands_os)):
                if np.array_equiv(conv_os, self.hands_os[k]):
                    idx_full_to_os_iso[idx] = k
                    break
            
            for s in range(4):
                
                if np.sum(conv_s[s][0]) == 2:
                    for k in range(len(hands_iso)):
                        if np.array_equiv(conv_s[s], hands_iso[k]):
                            idx_full_to_2s_iso[s][idx] = k
                            idx_full_to_1s_iso[s][idx] = k
                            break
                
                elif np.sum(conv_s[s][0]) == 1:
                    idx_full_to_2s_iso[s][idx] = idx_full_to_os_iso[idx]
                    for k in range(len(hands_iso)):
                        if np.array_equiv(conv_s[s], hands_iso[k]):
                            idx_full_to_1s_iso[s][idx] = k
                            break
                    
                else:
                    idx_full_to_2s_iso[s][idx] = idx_full_to_os_iso[idx]
                    idx_full_to_1s_iso[s][idx] = idx_full_to_os_iso[idx]
                    
            idx += 1
            
        translate_91_1326 = csr_matrix((91,1326),dtype=np.uint8)
        translate_169_1326 = [csr_matrix((169,1326),dtype=np.uint8) for _ in range(4)]
        translate_338_1326 = [csr_matrix((338,1326),dtype=np.uint8) for _ in range(4)]
        
        for i in range(len(idx_full_to_os_iso)):
            translate_91_1326[idx_full_to_os_iso[i], i] = 1
            
        for i in range(len(idx_full_to_2s_iso[0])):
            for s in range(4):
                translate_169_1326[s][idx_full_to_2s_iso[s][i], i] = 1            

        for i in range(len(idx_full_to_1s_iso[0])):
            for s in range(4):
                translate_338_1326[s][idx_full_to_1s_iso[s][i], i] = 1              
                
        return (idx_full_to_os_iso, idx_full_to_2s_iso, idx_full_to_1s_iso,
                translate_91_1326, translate_169_1326, translate_338_1326)
                        

                        
            
    def _get_rank_kickers(self):
        '''
        Builds lookup indexes for kicker strength so that overall hand
        strength can be calculated more efficiently

        '''
        rank_k_5 = {}
        rank_k_3 = {}
        rank_k_2_12 = {}
        rank_k_2_13 = {}
        
        idx = 0
        c1, c2, c3, c4, c5 = 0, 1, 2, 3, 4
        
        while idx < 1277:
            if c5 < 12:
                c5 += 1
            elif c4 < 11:
                c4 += 1
                c5 = c4 + 1
            elif c3 < 10:
                c3 += 1
                c4, c5 = c3 + 1, c3 + 2
            elif c2 < 9:
                if c2 == 8 and c1 == 0:
                    c1, c2, c3, c4, c5 = 1, 2, 3, 4, 6
                else:
                    c2 += 1
                    c3, c4, c5 = c2 + 1, c2 + 2, c2 + 3
            else:
                c1 += 1
                c2, c3, c4, c5 = c1 + 1, c1 + 2, c1 + 3, c1 + 5
            rank_k_5[np.array([c1,c2,c3,c4,c5],dtype=np.int8).tobytes()] = idx
            idx += 1
            
        c1, c2, c3, idx = 0, 1, 2, 0

        while c1 < 10:
            rank_k_3[np.array([c1,c2,c3],dtype=np.int8).tobytes()] = idx
            if c3 < 11:
                c3 += 1
            elif c2 < 10:
                c2 += 1
                c3 = c2 + 1
            else:
                c1 += 1
                c2, c3 = c1 + 1, c1 + 2
            idx += 1
            
        c1, c2, idx = 0, 1, 0      
        while c1 < 11:
            rank_k_2_12[np.array([c1,c2],dtype=np.int8).tobytes()] = idx
            if c2 < 11:
                c2 += 1
            else:
                c1 += 1
                c2 = c1 + 1
            idx += 1
            
            
        rank_k_2_13 = {}
        c1, c2, idx = 0, 1, 0  
        while c1 < 12:
            rank_k_2_13[np.array([c1,c2],dtype=np.int8).tobytes()] = idx
            if c2 < 12:
                c2 += 1
            else:
                c1 += 1
                c2 = c1 + 1
            idx += 1        
        
        return rank_k_5, rank_k_3, rank_k_2_12, rank_k_2_13
    
    def _convert_hand(self, hand, board_suit=-1):
        
        converted = np.zeros((2,13), dtype=np.int8)
        
        if board_suit in range(4):
            if hand[0] % 4 == board_suit:
                converted[0][hand[0] // 4] = 1
                
            if hand[1] % 4 == board_suit:
                converted[0][hand[1] // 4] = 1
        
        converted[1][hand[0] // 4] += 1
        converted[1][hand[1] // 4] += 1
        
        return converted
    
    def _convert_board(self, board):
        
        suit = -1
        conv = np.zeros((2,13), dtype=np.int8)
        
        for s in range(4):
            suited = np.count_nonzero([b % 4 == s for b in board])
            if suited >= 3:
                suit = s
                break
                
        for c in board:
            if c % 4 == suit:
                conv[0][c // 4] = 1
            conv[1][c // 4] += 1
        
        return conv, suit
    
    def _evals_to_util(self, evals):
        '''
        converts 1d array of hand evals (strength of hands) into 2d utilities 
        (win/loss) for hand i vs hand j as in terms of the chips bet: 1, 0, -1 
        represents win, tie, loss respectively
                                                        

        Parameters
        ----------
        evals : 1d array of hand evals
            

        Returns
        -------
        util : 2d np.ma_array of int8
        '''
        
        util = ma.ones((len(evals), len(evals)), dtype=np.int8)
        removed = np.where(evals == -1)[0]
        util[removed,:] = ma.masked
        util[:,removed] = ma.masked
        hands = np.argsort(evals)[len(removed):]
        better = []
        while len(hands) >= 1:
            ties = np.where(evals == evals[hands[0]])[0]
            util[np.ix_(ties,ties)] = 0
            try:
                util[np.ix_(ties, better)] = -1
            except:
                pass
            better += list(ties)
            hands = hands[len(ties):]
        return util
        
    
    def evaluate_hands(self, board):
        
        conv, suit = self._convert_board(board)
        
        if suit == -1:
            hands_ranks = self.lookup_os[conv.tobytes()]
            
        elif np.sum(conv[0] == 3):
            hands_ranks = self.lookup_3s[conv.tobytes()]
            
        else:
            hands_ranks = self.lookup_45s[conv.tobytes()]
            
            
    def _test_hand_evaluator(self):
        
        # enumerate all possible 5 card poker hands, returning
        # array with strength index of each hand
        
        all_evals = []
        for comb in combinations(range(0,52), 5):
            hand = np.zeros((2,13))
            for c in list(comb):
                hand[1][c % 13] += 1
            if max(comb) < 13 or (min(comb) >=13 and max(comb) < 26) or (min(comb) >=26 and max(comb) < 39) or min(comb) >= 39:
                hand[0] += hand[1]
            all_evals.append(self.evaluate_hand(hand))
        return all_evals
    
    def _normalize_rank(self, evals):
        '''
        Parameters
        ----------
        evals : np.ma_array
            DESCRIPTION.

        Returns
        -------
        norm_rank : np.ma_array dtype=uint8
            convert array with absolute hand strength evals (int16) into
            relative hand strength evals (int8). maps max strength value to a 
            number close to 255, and distributes evals across the range. this
            gives the normalized rank the property that a number gives a 
            (rough) approximation of strength of a hand in comparison to all
            other possible hole cards, for any converted board eval. eg a 
            normalized rank of 64 suggests that the hand is stronger than about
            3/4 of all starting hands.

        '''
    
        norm_rank = ma.empty(len(evals), dtype = np.uint8)
        norm_rank[np.where(evals == -1)[0]] = ma.masked
        
        current = 0
        i = 0
        
        while current < np.max(evals):
            next_rank = np.min(evals[np.where(evals > current)[0]])
            norm_rank[np.where(evals == next_rank)[0]] = i
            i += 1
            current = next_rank
            
        scale = np.uint8(255 // max(1,np.max(norm_rank)))
        norm_rank *= scale
    
        return norm_rank
    
    def get_hand_util_table(self, board):
        '''
        
        Parameters
        ----------
        board : board cards specified as list of length 5 in range(52)

        Returns
        -------
        utils : 1326x1326 masked array of utils (int8)
        util specifies whether hand i wins, loses, or ties vs hand j. 
        invalid hands (where a hand is blocked by board cards, or the two 
        hands block each other) are masked

        '''        
        converted = self._convert_board(board)
        board_eval = self.lookup[sum(converted[0][0])][converted[0].tobytes()]
        expanded_eval = ma.dot(board_eval, self.translate[len(board_eval)][converted[1]].toarray())            
        utils = self._evals_to_util(expanded_eval)
        
        self.hm.mask_blocked_hands(utils)
        for card in board:
            utils[self.hm.card_removal[card]] = ma.masked
            utils[:,self.hm.card_removal[card]] = ma.masked
        return utils
    
a = LookupTable()
z = a.get_hand_util_table(random.sample(range(52),5))