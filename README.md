# BrainDead 2K26 – ReelSense Solution
IIEST Shibpur Hackathon Submission

## Project Overview
This project implements an explainable and diversity-aware movie recommendation system using the MovieLens dataset.

The goal is not only predicting ratings but building a trustworthy recommender that:
- generates personalized recommendations
- explains why a movie is recommended
- reduces popularity bias
- increases diversity of suggestions

## Features
- Popularity baseline recommender
- User-User collaborative filtering
- Item-Item collaborative filtering
- Matrix factorization using SVD
- Hybrid recommendation model
- Natural language explanations
- Ranking evaluation metrics
- Diversity and novelty analysis

## Dataset
MovieLens Latest Small dataset by GroupLens Research.

## How to Run
1. Download dataset from:
   https://grouplens.org/datasets/movielens/

2. Extract `ml-latest-small.zip`

3. Open the notebook in Google Colab

4. Upload dataset folder

5. Run all cells sequentially

## Evaluation
The system is evaluated using:
- RMSE and MAE
- Precision@K and Recall@K
- NDCG and MAP
- Diversity and coverage metrics

## Conclusion
This system demonstrates that recommender systems should not only be accurate but also interpretable and diverse to improve user trust.
