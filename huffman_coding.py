class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(text):
    freq = Counter(text)
    heap = [Node(ch, f) for ch, f in freq.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)

    return heap[0]

def generate_codes(node, prefix="", code_map=None):
    if code_map is None:
        code_map = {}
    if node is None:
        return code_map
    if node.char is not None:
        code_map[node.char] = prefix or "0"
    generate_codes(node.left, prefix + "0", code_map)
    generate_codes(node.right, prefix + "1", code_map)
    return code_map

def huffman_encode(text):
    root = build_huffman_tree(text)
    codes = generate_codes(root)
    encoded = "".join(codes[ch] for ch in text)
    return encoded, codes, root

def huffman_decode(encoded, root):
    decoded = []
    curr = root
    for bit in encoded:
        curr = curr.left if bit == '0' else curr.right
        if curr.char is not None:
            decoded.append(curr.char)
            curr = root
    return "".join(decoded)


sample_text = "DIGITAL IMAGE PROCESSING"
encoded_str, huff_codes, huff_root = huffman_encode(sample_text)
decoded_str = huffman_decode(encoded_str, huff_root)

print("Huffman Codes:", huff_codes)
print("Encoded:", encoded_str)
print("Decoded:", decoded_str)
print(f"Original bits (8-bit ASCII): {len(sample_text)*8} | Encoded bits: {len(encoded_str)}")
