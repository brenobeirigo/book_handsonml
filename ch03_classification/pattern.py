# %%
from typing import Iterator
from collections import deque

def check(sequence, pattern, star):
    print(sequence, pattern, star)
    
    # Pattern is over but sequence not
    if not pattern and not sequence:
        return True
    
    seq = deque(sequence)
    pat = deque(pattern)
    
    
    while pat:
        p = pat.pop()
    
        while seq:
        
            c = seq.pop()
        
            print(c, "pattern", p,  p == '.')
            if p == c or p == '.':
                print("1", p, "==", c)
                if check(seq, pat, False):
                    return True

            elif p == '*':
                p = pat.pop()
                print("repeating", p)
                # Consume element previous to *
                while c == p:
                    c = seq.pop()
                    print("consuming", c)
                
                p = pat.pop()
                
                print("finish consume", seq, pat, p, c)
                                
                
                if p == c or p == '.':
                    print("2", p, "==", c)
                    if check(seq, pat, False):
                        return True

    return False

sequence = "daabbbbbbbbbbbbbbbbbbb"
pattern = "d..*"
check(sequence, pattern, False)
# %%
p = "p"
p[-1]
#%%
if p:
    print("p")
# %%
