# VECTOR DATABASES

## Introduction

A vector database indexes and stores vector embeddings for fast retrieval and similarity search, with capabilities like CRUD operations, metadata filtering, horizontal scaling and serverless.

Several GenAI applications (e.g., RAG) rely on vector embeddings, a type of vector representation that caries within it semantic information critical for the AI.
Vector DBs (e.g., Pinecone, Quadrant, Chroma, etc.) are specialize DBs to handle the embeddings data type. They offer optimized storage and querying capabilities for embeddings.
They have capabilities of traditional DB that are absent in standalone vector indexes and the specialization of dealing with embeddings absent in traditional DBs (i.e., scalar-based databases).
To build data into a VDB an embedding model is needed, in order to project content (e.g., text) onto an high-dimensional vector space, then the embedding with its related original content is inserted into the VDB.
When a query is issued the query embedding is generated and via similarity search the most similar vectors/contents are retrieved.
Standalone vector indices (e.g., FAISS, NMSLIB, Lucene, etc.) improve search and retrieval of embeddings but lack capabilities common to DBs.
Whereas, VDBs are purpose-built to manage vector embeddings, thus giving advantages over them.

The main advantages are:

- Data management: Vector databases offer well-known and easy-to-use features for data storage, like inserting, deleting, and updating data. This makes managing and maintaining vector data easier than using a standalone vector index like FAISS, which requires additional work to integrate with a storage solution.

- Metadata storage and filtering: Vector databases can store metadata associated with each vector entry. Users can then query the database using additional metadata filters for finer-grained queries.

- Scalability: Vector databases are designed to scale with growing data volumes and user demands, providing better support for distributed and parallel processing. Standalone vector indices may require custom solutions to achieve similar levels of scalability (such as deploying and managing them on Kubernetes clusters or other similar systems). Modern vector databases also use serverless architectures to optimize cost at scale.

- Real-time updates: Vector databases often support real-time data updates, allowing for dynamic changes to the data to keep results fresh, whereas standalone vector indexes may require a full re-indexing process to incorporate new data, which can be time-consuming and computationally expensive. Advanced vector databases can use performance upgrades available via index rebuilds while maintaining freshness.

- Backups and collections: Vector databases handle the routine operation of backing up all the data stored in the database. Pinecone also allows users to selectively choose specific indexes that can be backed up in the form of “collections,” which store the data in that index for later use.

- Ecosystem integration: Vector databases can more easily integrate with other components of a data processing ecosystem, such as ETL pipelines (like Spark), analytics tools (like Tableau and Segment), and visualization platforms (like Grafana) – streamlining the data management workflow. It also enables easy integration with other AI related tooling like LangChain, LlamaIndex, Cohere, and many others.

- Data security and access control: Vector databases typically offer built-in data security features and access control mechanisms to protect sensitive information, which may not be available in standalone vector index solutions. Multitenancy through namespaces allows users to partition their indexes fully and even create fully isolated partitions within their own index.

The main steps of a VDB are:

- Indexing: The vector representation of the content is mapped to a data structure that enables the retrieval

- Querying: The vector DB compares the indexed query vector with the indexed embeddings (KB) to find the nearest neighbors via a similarity metric

- Post Processing: The VDB can post-process the search results like filtering or re-ranking

Traditional DBs store scalar values (e.g., integers, strings, floats, booleans, etc.) into rows and columns, whereas VDBs store vector embeddings.
Therefore, the querying of data is different between the two of them. VDBs need similarity search algorithms to get the most similar contents to a query.
The main backbone of similarity search for vector embeddings are KNN/ANN alogorithms.
They returns the top K neighbors of a query, with the usual trade-off between accuracy and latency/speed. The more accurate the retrieval is the more time it needs.
Usually brute-force KNN is not suggested as retrieval method in semantic search. This is because it compares the query vector with all the content in the VDB, leading to an higher latency at inference time. Therefore, Approximate KNN (ANN) is used to reduce retrieval time but preserving good accuracy.
To allow ANN some data structures that can be traversed quickly are created when indexing vectors.
Indexing can be defined as the process of smartly storing vector embeddings to optimize the retrieaval process.
Indeed, via indexing the arrangement of vector embeddings is done in a way that similar vectors are somehow grouped together.
Using KNN (brute-force search) leads to a linear time complexity $O(n)$ when comparing a query vector with all the vectors in the database.
The purpose of ANN, achieved via indexing algorithms, is to reduce the complexity to achieve a sub-linear one.
This is done by using the index (data structure) to reduce the scope of the search to just a subset of the vectors (potential candidates).

The main indexing techniques are:

- Flat Index

- Inverted File Index (IVF)

- Hierarchical Navigable Small Worlds (HNSW)

- Locality Sensitive Hashing (LSH)

Before talking about indexing it's important to go over some more concepts like product quantization (PQ) and scalar quantization (SQ).

Vector similarity search can require huge amounts of memory. Indexes containing 1M dense vectors (a small dataset in today’s world) will often require several GBs of memory to store.
The problem of excessive memory usage is exasperated by high-dimensional data, and with ever-increasing dataset sizes, this can very quickly become unmanageable.
Product quantization (PQ) is a popular method for dramatically compressing high-dimensional vectors largely reducing their memory footprint and speeding up the similarity search by exploiting the compressed vectors instead of the raw ones.

Quantization is a generic method that refers to the compression of data into a smaller space.
Quantization is different from dimensionality reduction. Given a vector with dimension $D$ (e.g., 1024) made of 32-bit floats (the scope $S$), dimensionality reduction generates another representation of the raw vector by reducing its dimension $D$ (i.e., $D_{reduced} < D$).
On the other hand, quantization compresses the vector by reducing the scope $S$ rather  than the dimension $D$.

There are many ways of doing this. For example clustering. When clustering a set of vectors the larger scope of potential values (all possible vectors) is replaced with a smaller discrete and symbolic set of centroids.
And this is really the definition in a nutshell of a quantization operation. The transformation of a vector into a space with a finite number of possible values, where those values are symbolic representations of the original vector.

Product quantization (PQ) is a lossy-compression method that compresses the vectors via a three step process:

- It splits each high-dimensional vector into a fixed-equally-sized sub-vectors/chunks.

- For each set of sub-vectors (subspace) a clustering operation is performed, creating multiple centroids for each subspace

- Assigning each of these subvectors to its nearest centroid (also called reproduction/reconstruction value).

- Replace each centroid values with an unique IDs, each representing a centroid.

- Create, for each original high-dimensional vector, a tiny vector of centroid IDs.

At the end of the process the original vector is mapped to a smaller vector of IDs that requires way less memory.

PQ is a compression technique that reduces the dimensionality of the vectors.
Therefore, PQ speeds up the similarity computation becuase it reduces the dimensionality of the embedding vectors and query vectors, but it also reduces the memory footprint of the stored vectors.

Scalar quantization (SQ) is a lossy-compression method that compresses the vectors scope.
SQ takes each dimension of the vector and buckets them into some smaller data type, for instance from float32 to int8.
However, SQ  does not justs rounds the data type but it applies a numerical transformation to the data.

The equations used in Lucene for instance are:

$
int8 \approx \frac{127}{max - min} * (float32 - min)
$

$
float32 \approx \frac{max - min}{127} * (int8 + min)
$

## Flat Index

A flat index is by and large the most basic indexing strategy, but arguably also the most overlooked. With flat indexing, the query vector is compared with every other vector in the database. Therefore, the index is simply a flat data structure which is exactly the size of the dataset.
Given that in flat index there is no approximation, it gives back the most accurate results but with a slower fashion.
Put in another perspective flat index scales linearly with the size of the dataset, indeed it's characterized by linear time complexity $O(n)$.
Flat index does an exhaustive search of all the vectors in the database to find the nearest neighbors with the query one, thus it computes the similarity between the query vector and all the vectors in the database.
Therefore, the flat indexing is a good choice when the dataset is small because it does not scale well with the size of the dataset.
The flat index can be combined with Product Quantization to reduce the memory footprint of the vectors, this leads to a kind of composite index.

## Inverted File Index (IVF)

The IVF technique is an indexing method that allows for approximate search as opposed to flat index that allows for exact search. Put into another perspective IVF is an approximate nearest neighbor search algorithm that uses a clustering technique to reduce the search space.
The IVF partitions the index (i.e., dataset) into Voronoi cells (clusters) each of them with their own centroid, thus applying a clustering algorithm (e.g., K-means) to the vector embeddings.
Each vector embedding is assigned to the cluster that has the closest centroid, this allows the system not to traverse the entire index to find the nearest neighbors.
Indeed, when a query vector is given the system finds the closest centroid to the query vector and then it performs a (brute-force) search within that cluster of vector embeddings. The IVF needs parameters like the number of clusters/cells (usually called nlist) to partition the index into and the number of clusters/cells to search into (usually called nprobe). The higher the number of clusters to search into, the more time it takes but likely with better accuracy.
IVF is able to speed up the search by reducing the scope of the search space.
Unlike flat index, IVF needs the index to be trained in order to create clusters and centroids.

>**Info:** The main idea of the Voronoi cells (also called Dirichlet tessellation) is to build non intersecting regions where each data point is assigned. Moreover, the property of Voronoi cells is that the distance between the cell centroid an a point of the cell is less than the distance of that point with any other cell centroid.

There are several variants of the IVF.

IVF-Flat, this is the most basic variant of IVF. It is a simple inverted file index that uses a flat index to store the vectors in each cluster. Therefore, when a query vector is given the system finds the closest centroid/s to the query vector and then it performs a (brute-force) search within that cluster/s of vector embeddings.
It's a good balance between accuracy and speed, especially in smalle/medimum-sized datasets where high accuracy is required.

IVF-PQ, this is a variant of IVF that uses product quantization to compress the vectors in each cluster.
When a query vector is given, the system finds the closest centroid/s and perform search within the cluster/s by comparing the compressed query and the compressed vectors in the cluster/s.
Therefore, the IVF-PQ exploits the quicker search due to the restricted search space and it also leads to even faster computation due to a faster similarity calculation due to the compression of vectors. However, PQ is an additional approximation step affecting the similarity calculation.
Therefore, IVF-PQ is a good choice when the dataset is large and the accuracy is not critical becuase it leads to two differnt sources of approximations.

IVF-SQ, this variant of IVF uses scalar quantization to compress the vectors in each cluster.

## Hierarchical Navigable Small Worlds (HNSW)

Hierarchical Navigable Small Wolrd (HNSW) graphs are among the top-performing indexes for vector similarity search.

HNSW is a graph-based indexing technique that is designed to efficiently search for nearest neighbors in high-dimensional spaces.
Its graph-like structure takes inspiration from the probability skip list and the Navigable Small World (NSW).

The skip list is a data structure that combines the quick insertion of a linked list and the fast retrieval of an array. It's based on a multilayer architecture where data is organized accross multiple layers, with each layer containing a subset of the data.
Navigable Small World (NSW) is similar to a proximate graph where nodes are linked together based on how similar they are to each other. The greedy method is used to search for the nearest neighbors.

The HNSW creates a hierarchical, tree-like structure where each node of the tree represents a set of vectors. The edges between the nodes represent the similarity between the vectors. The algorithm starts by creating a set of nodes, each with a small number of vectors. Tis can be done randomly or by clustering the vectors with algorithms like K-means, where each cluster becomes a node.
The algorithm examines the vectors of each node and draws an edge between that node and the nodes that have the most similar vectors to the one it has.
When one queries an HNSW index, it uses this graph to navigate through the tree, visiting the nodes that are most likely to contain the closest vectors to the query vector.

HNSW can be combined with scalar quantization, thus getting the HNSW-SQ index.
HNSW has a time complexity of $O(nlog(n))$ for insertion and $O(log(n))$ for search, where $n$ is the number of vectors in the index.

## Locality Sensitive Hashing (LSH)

Locality Sensitive Hashing (LSH) is a technique used to approximate nearest neighbors in high-dimensional spaces.
LSH maps similar vectors into hash tables ("buckets"), using a set of hashing functions.

To find the nearest neighbors for a given query vector, we use the same hashing functions used to “bucket” similar vectors into hash tables. The query vector is hashed to a particular table and then compared with the other vectors in that same table to find the closest matches. This method is much faster than searching through the entire dataset because there are far fewer vectors in each hash table than in the whole space.

It’s important to remember that LSH is an approximate method, and the quality of the approximation depends on the properties of the hash functions. In general, the more hash functions used, the better the approximation quality will be. However, using a large number of hash functions can be computationally expensive and may not be feasible for large datasets.
LSH is an algorithm characterized by sub-linear time complexity.

The traditional LSH algorithm is based on a three step process (there are other methods like Random Projection):

1. Shingling: It is the process of converting a string of text into a set of ‘shingles’. The process is similar to moving a window of length k down our string of text and taking a picture at each step. We collate all of those pictures to create our set of shingles. Shingling removes duplicates (indeed it creates a set of shingles). Then a sparse vector is created. To do this first all the shingles are pooled into a single set called vocabulary. Then a vector of zeros of length equal to the vocabulary, then for every shingle appearing in the set and in the zero vector a one is placed in the position of the shingle in the vocabulary. This is a sparse vector created via a one-hot encoding.

2. MinHashing: It converts the sparse vector into a smaller dense one. A random minhash function is generated for every position of the dense vector. The minhash function is just a random order of numbers from 1 to the length of the vocabulary.

3. Band and Hash: It takes the signature/dense vectors, hashing segments of each signature and looks for hash collisions. The outptu are vectors of equal length that contains positive integers from one to length of vocabulary. These are then passed to the hash function.


## Similarity Search

Similarity search can be used to compare data quickly. Given a query that can be image, text, video, the similarity search returns relevant results.
In vector similarity search, that is a type of similarity search, a vector representations capturing the semantic meaning of the data are indexed and stored.
There are different types of indexes, one needs to choose based on factors such as dataset size, search frequency, or search-quality vs. search-speed.

The flat index is the easiest one becuause it does not apply any operation to the vectors. Therefore, in flat index there is no training and no parameters to set/optimize.
Because there is no approximation or clustering of vectors — these indexes produce the most accurate results. There is perfect search quality, but this comes at the cost of significant search times.
Therefore with a flat index, given a query vector it's compared using a similarity metric  with all the vectors in the indes.
After computing all the similarities, the results are sorted by the similarity metric and the top-k results are returned.
In other terms, flat index performs an exhaustive search via KNN (also referred as brute-foce KNN).
Flat index is a good choice when the dataset is small, the search frequency is low and the search quality is high priority.

>**Info:** there are several similarity metrics and vector index, in FAISS it has been shown that for larger datasets, the L2 distance is slower than using the inner product.

To speed up the search time/inference, which is crucial for large datasets or low latency applications, there are mainly macro possibilities:

- Reduce vector size — through dimensionality reduction or reducing the number of bits representing our vectors values.

- Reduce search scope — viaclustering or organizing vectors into tree structures based on certain attributes, similarity, or distance — and restricting our search to closest clusters or filter through most similar branches.

Using either of these approaches means that we are no longer performing an exhaustive nearest-neighbors search but an approximate nearest-neighbors (ANN) search — as we no longer search the entire, full-resolution dataset.

To reduce the search scope by creating a data structure that can be used to restrict the search space, there are different approaches.

Locality Sensitive Hashing (LSH) works by grouping vectors into buckets by processing each vector through a hash function that maximizes hashing collisions — rather than minimizing as is usual with hashing functions.
Hashing collision is where two different objects (keys) produce the same hash value.

The logic of hashing collision is reverted in LSH because LSH is meant to group similar objects together. When a new query object (or vector) is introduced, the LSH algorithm can be used to find the closest matching groups.

The main parameter of LSH is the nbits, which stands for the 'resolution' of the hashed vector.
The higher the nbits the higher the search accuracy, but at the cost of more memory and higher latency/search time.
The nbits is paired with the dimension of the vectors, therefore LSH suffers from the curse of dimensionality.
Indeed, LSH allows to get quick search with good accuracy but it's ideal for low dimensional vectors or small datasets.

Hierarchical Navigable Small World (HNSW) graphs are another, more recent development in search. HNSW-based ANNS consistently top out as the highest performing indexes.
HNSW is a further adaption of navigable small world (NSW) graphs using probability skip list — where an NSW graph is a graph structure containing vertices connected by edges to their nearest neighbors.
The ‘NSW’ part is due to vertices within these graphs all having a very short average path length to all other vertices within the graph — despite not being directly connected.
At a high level, HNSW graphs are built by taking NSW graphs and breaking them apart into multiple layers. With each incremental layer eliminating intermediate connections between vertices.
For bigger datasets with higher-dimensionality — HNSW graphs are some of the best performing indexes possible.
The HNSW has three parameters:

- M: The number of nearest neighbors that each vertex will connect to.

- efSearch: How many entry points will be explored between layers during the search.

- efConstruction: How many entry points will be explored when building the index.

M and efSearch are positively related with search accuracy, therefore the higher the more accurate is the search. However, as the usual trade-off, higher M and efSearch values will also increase the search time.
Whereas, efConstruction mainly affects the index building time in a positive fashion. However, for high values of M and high query volume the efConstruction can affect even search time (and not only building time).

HNSW gives great search-quality at very fast search-speeds — but there’s always a catch — HNSW indexes take up a significant amount of memory. The increase in memory footprint is due to the M parameter and not the efSearch and efConstruction parameters.
So, where RAM is not a limiting factor HNSW is great as a well-balanced index that one can push to focus more towards quality by increasing the three parameters.

The Inverted File Index (IVF) index consists of search scope reduction through clustering. It’s a very popular index as it’s easy to use, with high search-quality and reasonable search-speed.
It works on the concept of Voronoi diagrams — also called Dirichlet tessellation (a much cooler name).
To understand Voronoi diagrams, we need to imagine the highly-dimensional vectors placed into a 2D space. Then place a few additional points in the 2D space, which will become the ‘cluster’ (Voronoi cells in our case) centroids.
Then extend an equal radius out from each of the centroids. At some point, the circumferences of each cell circle will collide with another — creating the cell edges.
Now, every datapoint will be contained within a cell — and be assigned to that respective centroid.
Just as with other indexes, the query vector is introduced — this query vector must land within one of the cells, at which point it restricts the search scope to that single cell.
But there is a problem if the query vector lands near the edge of a cell — there’s a good chance that its closest other datapoint is contained within a neighboring cell. This is the edge problem.
To mitigate this issue and increase search-quality is increase an index parameter known as the nprobe value. With nprobe one can set the number of neighbor cells to search.
The higher the nprobe the higher the accuracy of the search, but at the cost of higher search time.
Therefore, the two main parameters of IVF are:

- nprobe: The number of cells (clusters) to search.

- nlist: The number of cells (clusters) to create.

A higher nlist means comparing the query vector to more centroid vectors — but after selecting the nearest centroid’s cells to search, there will be fewer vectors within each cell. So, increase nlist to prioritize search-speed.
As for nprobe, that's tje opposite. Increasing nprobe increases the search scope — thus prioritizing search-quality but at the cost of higher search time.
In terms of memory, IVF is a memory-efficient index. Therefore, changing nprobe is not affecting the memory usage and chaning nlist is slightly affecting the memory usage. Indeed, a larger nlist means a marginally more memory requirements.