## 🐍 faiss01.py

import numpy as np
import faiss

# Step 1: Generate sample data (10 vectors, 5 dimensions)
np.random.seed(42)
data = np.random.random((10, 5)).astype('float32')

# Step 2: Create FAISS index (L2 distance)
index = faiss.IndexFlatL2(5)  # dimension = 5

# Step 3: Add vectors to index
index.add(data)
print(f"Number of vectors in index: {index.ntotal}")

# Step 4: Perform a similarity search
query = np.random.random((1, 5)).astype('float32')
distances, indices = index.search(query, k=3)  # top-3 neighbors

print("Query vector:", query)
print("Nearest neighbors (indices):", indices)
print("Distances:", distances)
