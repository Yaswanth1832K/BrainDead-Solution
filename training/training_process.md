# Training Process

The recommendation system follows the pipeline below:

1. Data Preprocessing
   - Convert timestamp to datetime
   - Clean movie genres
   - Clean user tags
   - Construct user-item interaction matrix
   - Create genre feature vectors

2. Time-Based Train-Test Split
   Instead of a random split, we use a leave-last-interaction strategy.
   For each user:
   - The most recent rating is used as the test sample
   - Previous ratings are used for training

   This simulates real-world recommendation where we predict a user's future behavior.

3. Model Training
   - Train SVD matrix factorization model on training data
   - Learn latent embeddings for users and movies

4. Recommendation Generation
   - Generate Top-K personalized recommendations
   - Combine collaborative and content features (hybrid model)

5. Explainability
   - Generate natural language explanation using user history and genre overlap
