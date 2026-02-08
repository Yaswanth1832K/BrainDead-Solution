# Evaluation Metrics

We evaluate the recommender system using multiple categories of metrics.

A. Rating Prediction Metrics
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)

These measure how accurately the model predicts ratings.

B. Ranking Quality Metrics
- Precision@K
- Recall@K
- NDCG@K
- MAP@K

These measure how well the recommended Top-K movies match user preferences.

Note:
We use a time-based evaluation protocol. Each user has only one future interaction in the test set. Therefore ranking metrics may appear lower but represent realistic performance.

C. Diversity and Novelty Metrics
- Intra-List Diversity
- Catalog Coverage
- Popularity-Normalized Hits
- Novelty Score

These ensure the recommender avoids popularity bias and provides varied suggestions.

The goal of the system is not only accuracy but also explainability and trustworthy recommendations.
