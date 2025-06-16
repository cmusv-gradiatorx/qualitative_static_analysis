# Clustering Report

Three clustering strategies on task 4 Gilded Rose Kata: pure java code embedding, repomix embedding, issue-based embedding.
Three clustering algorithms are applied: kmeans, hierarchical, and DBSCAN.

### Java Code Embedding
Experience starCoder2:3b (2048 dim) and starCoder2:7b (4096 dim) models.

### Repomix Embedding
Before applying starCoder models, run through existing repomix to process the codebase.

### Issue Embedding

**Process Flow:**
1. **Data Loading**: Load student issues from JSON file containing LLM autograder outputs
2. **Issue Clustering**: Group semantically similar issues using sentence transformers (consine similarity threshold: 0.7)
3. **Vocabulary Creation**: Build vocabulary from issue clusters or individual issues
4. **Student Embedding Generation**:
   - **Semantic Component**: Join student's issues then feed to sentence transformer (all-MiniLM-L6-v2 384 dim) embedding
   - **Profile Component**: Binary vector indicating presence/absence of each vocabulary item
   - **Final Embedding**: Concatenate [semantic_embedding × 0.2 + profile_embedding × 0.8]

### KMeans
Specify the number of clusters to the number of unique grade students were given in last Fall.

### Hierarchical
Specify the number of clusters to the number of unique grade students were given in last Fall.

### DBSCAN
Can eleminate outlier. In Gilded Rose there was only one student given 0.25, the rest all have clusters. Tuned `eps` to eliminate that students. For issue `eps = 30`; for code based `eps = 100`

## Result
**Note: before asking Prof Leo to manually go over clustering, I used final grades students received to identify clusters.**
1. DBSCAN works well on eliminating outlier (students who received very different grade than others). But because of the large dimentional space, the data point layed in sparse. The alogrithm results in just one big clusters. After tried both code embedder and issue embedder, noticed that DBSCAN does not fit in this scenario.
2. KMeans works well on clustering, but the cluster results is not quite right. For both codebase and issues embedder, it would cluster students with different scores into same group, resulting a mis-cluster rate about 45~60%. 
3. Hierarchical with issue based embedder turns out to work the best among these (about 40% mis cluster). But this is promising. To improve this, we can improve the issues vocabulary creation process to have category and serverity. We can also tuned the weight of semantic component and profile component. However, hierarchical approach has issues as well. The biggest issue is that the Silhouette score is only 0.154, indicating that the clustering is week and points are near the boundary of clusters.

## Additional Work
Refactored previous 3 weeks work to have factory design pattern, allowing future extensibility.

---
**Report Written on:** June 15, 2025