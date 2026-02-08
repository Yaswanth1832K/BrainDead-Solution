# Model Description

This project implements multiple recommendation strategies as required in the ReelSense challenge.

1. Popularity-Based Recommender
   Recommends the highest-rated movies overall. This serves as a baseline.

2. User-User Collaborative Filtering
   Finds users with similar rating behavior using cosine similarity and recommends movies liked by similar users.

3. Item-Item Collaborative Filtering
   Finds movies similar to a given movie based on rating patterns.

4. Matrix Factorization (SVD)
   We apply Singular Value Decomposition using the Surprise library to learn latent user and movie factors.
   The model learns hidden preferences of users and hidden characteristics of movies.

5. Hybrid Recommendation Model
   Combines collaborative filtering predictions with genre-based content similarity.
   This improves recommendation quality and reduces popularity bias.

Explainability:
Recommendations are accompanied by a natural language explanation based on the user's highly rated movies and shared genres.
