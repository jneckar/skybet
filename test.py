CARD_RANKS = ['A','K','Q','J','T','9','8','7','6','5','4','3','2']
CARD_SUITS = ['s','h','d','c']
DECK = []
HANDS = []

for r in CARD_RANKS:
    for s in CARD_SUITS:
        DECK.append(r+s)

for r1 in range(13):
    for r2 in range(r1, 13):
        
        for s1 in range(4):
            for s2 in range(s1, 4):
                
                if not (r1 == r2 and s1 == s2):
                    
                    i = r1 * 4 + s1
                    j = r2 * 4 + s2
                    
                    HANDS.append(DECK[i]+DECK[j])