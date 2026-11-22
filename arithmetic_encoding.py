getcontext().prec = 50  # High precision for longer strings

def get_probability_ranges(text):
    total = Decimal(len(text))
    freq = Counter(text)
    ranges = {}
    low = Decimal(0)
    for ch in sorted(freq.keys()):
        prob = Decimal(freq[ch]) / total
        ranges[ch] = (low, low + prob)
        low += prob
    return ranges

def arithmetic_encode(text):
    ranges = get_probability_ranges(text)
    low, high = Decimal(0), Decimal(1)

    for ch in text:
        rng = high - low
        ch_low, ch_high = ranges[ch]
        high = low + rng * ch_high
        low = low + rng * ch_low

    # Any number in [low, high) represents the encoded message; midpoint is safest
    code = (low + high) / Decimal(2)
    return code, ranges

def arithmetic_decode(code, length, ranges):
    decoded = []
    low, high = Decimal(0), Decimal(1)

    for _ in range(length):
        rng = high - low
        scaled = (code - low) / rng
        for ch, (ch_low, ch_high) in ranges.items():
            if ch_low <= scaled < ch_high:
                decoded.append(ch)
                high = low + rng * ch_high
                low = low + rng * ch_low
                break
    return "".join(decoded)

# --- Test ---
msg = "ARITHMETIC"
code_val, prob_ranges = arithmetic_encode(msg)
decoded_msg = arithmetic_decode(code_val, len(msg), prob_ranges)

print("Original:", msg)
print("Encoded Value:", code_val)
print("Decoded:", decoded_msg)
