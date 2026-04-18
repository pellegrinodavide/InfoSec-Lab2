import numpy as np

# ── [7,4,3] Hamming code codewords ───────────────────────────────────────────

CODEBOOK = [
    '0000000', '1000110', '0100101', '1100011',
    '0010011', '1010101', '0110110', '1110000',
    '0001111', '1001001', '0101010', '1101100',
    '0011100', '1011010', '0111001', '1111111'
]

# ── random binning encoder ────────────────────────────────────────────────────

class RandomBinningEncoder:
    """
    Message d ∈ {0,1}^3  →  bin T_{x|u}(d) = { codeword starting with 0‖d,
                                                  codeword starting with 1‖d̄ }
    x is chosen uniformly at random from the 2-codeword bin.
    """
    def __init__(self, codebook=CODEBOOK, rng=None):
        self.codebook = codebook
        self.rng = rng or np.random.default_rng(42)
        self.bins = self._build_bins()

    def _build_bins(self):
        bins = {}
        for d in range(8):                          # d ∈ {0,1}^3
            d_bits  = format(d, '03b')              # e.g. '110'
            d_comp  = ''.join('1' if b=='0' else '0' for b in d_bits)  # d̄
            prefix0 = '0' + d_bits                  # 0‖d
            prefix1 = '1' + d_comp                  # 1‖d̄
            bin_cw  = [c for c in self.codebook
                       if c[:4] == prefix0 or c[:4] == prefix1]
            bins[d_bits] = bin_cw
        return bins

    def encode(self, d: str) -> str:
        """Pick one codeword uniformly at random from the bin for message d."""
        return self.rng.choice(self.bins[d])

# ── verify ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    enc = RandomBinningEncoder()

    print("Bins:")
    for d, cws in enc.bins.items():
        print(f"  d = {d}  →  {cws}")

    print("\nSample encodings (5 per message):")
    for d in [format(i, '03b') for i in range(8)]:
        samples = [enc.encode(d) for _ in range(5)]
        #print(f"  d = {d}  →  {samples}")
        print(f"  d = {d}  →  {[str(s) for s in samples]}")