## Movie Recommender System

### Description

This is a movie recommender system based on [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset). For local use please place the `.csv` files into `/data/` directory. Alternatively you can use `PostgresDataLoader` from `data_loader.py` to read the data from a PostgreSQL database.

Three different approaches are implemented:
- Demographic filtering
- Content-based filtering
- Collaborative filtering