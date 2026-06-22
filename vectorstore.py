import struct
import math

class VectorStore:
    def __init__(self, filename="vectors.bin"):
        self.filename = filename
        self.vectors = []
        self.metadata = []
    
    def add_vector(self, vector_data, label=""):
        """Store a vector with a label"""
        packed = struct.pack('f' * len(vector_data), *vector_data)
        self.vectors.append(packed)
        self.metadata.append(label)
    
    def cosine_similarity(self, v1, v2):
        """Calculate similarity between two vectors"""
        dot = sum(a * b for a, b in zip(v1, v2))
        mag1 = math.sqrt(sum(a * a for a in v1))
        mag2 = math.sqrt(sum(b * b for b in v2))
        return dot / (mag1 * mag2) if mag1 * mag2 != 0 else 0
    
    def search(self, query_vector, top_k=2):
        """Find most similar vectors"""
        scores = []
        for i, packed in enumerate(self.vectors):
            # Unpack back to floats
            unpacked = struct.unpack('f' * (len(packed) // 4), packed)
            sim = self.cosine_similarity(query_vector, unpacked)
            scores.append((sim, self.metadata[i], i))
        
        scores.sort(reverse=True)
        return scores[:top_k]

# Test it
if __name__ == "__main__":
    store = VectorStore()
    store.add_vector([1.0, 2.5, 3.14, 0.5], "vector_1")
    store.add_vector([1.1, 2.4, 3.15, 0.51], "vector_2")
    store.add_vector([5.0, 1.0, 0.5, 2.0], "vector_3")

    # Search for similar to first one
    results = store.search([1.0, 2.5, 3.14, 0.5], top_k=2)
    print("Search results:")
    for sim, label, idx in results:
        print(f"  {label}: similarity = {sim:.4f}")