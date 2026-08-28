import struct
import math
from collections import defaultdict

class VectorStore:
    def __init__(self, filename="vectors.bin"):
        self.filename = filename
        self.vectors = []
        self.metadata = []
        self.min_val = None
        self.max_val = None
    
    def add_vector(self, vector_data, label=""):
        """Store a vector as int8 (quantized)"""
        if self.min_val is None:
            self.min_val = min(vector_data)
            self.max_val = max(vector_data)
        else:
            self.min_val = min(self.min_val, min(vector_data))
            self.max_val = max(self.max_val, max(vector_data))
        
        quantized = []
        for val in vector_data:
            scaled = (val - self.min_val) / (self.max_val - self.min_val + 1e-6)
            int_val = int(scaled * 255)
            quantized.append(int_val)
        
        packed = struct.pack('B' * len(quantized), *quantized)
        self.vectors.append(packed)
        self.metadata.append(label)
    
    def dequantize(self, packed_vector):
        """Convert int8 back to floats"""
        unpacked = struct.unpack('B' * len(packed_vector), packed_vector)
        result = []
        for val in unpacked:
            scaled = val / 255.0
            original = scaled * (self.max_val - self.min_val) + self.min_val
            result.append(original)
        return result
    
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
            unpacked = self.dequantize(packed)
            sim = self.cosine_similarity(query_vector, unpacked)
            scores.append((sim, self.metadata[i], i))
        
        scores.sort(reverse=True)
        return scores[:top_k]
    
    def simple_clustering(self, threshold=0.95):
        """Group similar vectors (cluster them)"""
        clusters = defaultdict(list)
        cluster_id = 0
        processed = set()
        
        for i, packed_i in enumerate(self.vectors):
            if i in processed:
                continue
            
            vector_i = self.dequantize(packed_i)
            cluster = [i]
            processed.add(i)
            
            # Find all similar vectors
            for j, packed_j in enumerate(self.vectors):
                if j in processed or j <= i:
                    continue
                
                vector_j = self.dequantize(packed_j)
                sim = self.cosine_similarity(vector_i, vector_j)
                
                if sim >= threshold:
                    cluster.append(j)
                    processed.add(j)
            
            clusters[cluster_id] = cluster
            cluster_id += 1
        
        return clusters
    
    def merge_clusters(self, clusters):
        """Merge similar vectors, keep only one per cluster"""
        merged_vectors = []
        merged_metadata = []
        
        for cluster_id, indices in clusters.items():
            if not indices:
                continue
            
            # Keep first vector, discard others
            merged_vectors.append(self.vectors[indices[0]])
            merged_metadata.append(f"{self.metadata[indices[0]]} (merged {len(indices)} vectors)")
        
        self.vectors = merged_vectors
        self.metadata = merged_metadata
        
        return len(merged_vectors)

# Test it
if __name__ == "__main__":
    store = VectorStore()
    store.add_vector([1.0, 2.5, 3.14, 0.5], "memory_1")
    store.add_vector([1.05, 2.48, 3.16, 0.52], "memory_2")  # Very similar
    store.add_vector([1.02, 2.49, 3.15, 0.51], "memory_3")  # Very similar
    store.add_vector([5.0, 1.0, 0.5, 2.0], "memory_4")      # Different
    
    print("Before clustering:")
    print(f"  Total vectors: {len(store.vectors)}")
    
    clusters = store.simple_clustering(threshold=0.95)
    print(f"\nClusters found: {len(clusters)}")
    for cid, indices in clusters.items():
        print(f"  Cluster {cid}: {len(indices)} vectors")
    
    merged_count = store.merge_clusters(clusters)
    print(f"\nAfter merging:")
    print(f"  Total vectors: {merged_count}")
    print(f"  Space saved: {len(store.metadata)} → {merged_count}")
    
    # Search still works
    results = store.search([1.0, 2.5, 3.14, 0.5], top_k=2)
    print(f"\nSearch results after clustering:")
    for sim, label, idx in results:
        print(f"  {label}: similarity = {sim:.4f}")