from Task1 import WiretapUniformErrorChannel
from Task2 import RandomBinningEncoder

import numpy as np

def hamming_distance(s1, s2):
    """Calculate the number of positions at which the symbols are different."""
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

class RandomBinningDecoder:
    def __init__(self, codebook):
        self.codebook = codebook

    def decode(self, y: str) -> str:
        hat_x = min(self.codebook, key=lambda a: hamming_distance(y, a))
        x1 = hat_x[0]
        hat_d = hat_x[1:4]
        if x1 == '0':
            hat_u = hat_d
        else:
            hat_u = ''.join('1' if b == '0' else '0' for b in hat_d)
            
        return hat_u
    
if __name__ == "__main__":
    enc = RandomBinningEncoder()
    dec = RandomBinningDecoder(enc.codebook)

    print("Testing encoder and decoder:")
    for d in [format(i, '03b') for i in range(8)]:
        encoded = enc.encode(d)
        decoded = dec.decode(encoded)
        print(f"  Message: {d}  →  Encoded: {encoded}  →  Decoded: {decoded}")
        
    print("\n--------------------------------------------------------------\n")
        
    print("Testing encoder and decoder through legitimate channel noise:")
    ch = WiretapUniformErrorChannel(n=7, r=1, s=3)
    for d in [format(i, '03b') for i in range(8)]:
        encoded = enc.encode(d)
        y, _ = ch.transmit(int(encoded, 2), N=1)
        y_str = format(y[0], '07b')
        decoded = dec.decode(y_str)
        print(f"  Message: {d}  →  Encoded: {encoded}  →  Received: {y_str}  →  Decoded: {decoded}")  
    