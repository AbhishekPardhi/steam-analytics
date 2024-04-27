# steam-analytics

We are exploring trends in Steam Games to learn what will be an ideal strategy for:

- Improving reach
- User retention and satisfaction with updates and seasonal updates
- User experience enhancement through Steam game data analysis
- Difficulty and price-to-profit analysis

The goal is to align the development and improvement of games with player preferences, ultimately improving user satisfaction, retention, and overall success on the Steam platform.

## How to run the Webapp

First, create an environment and install all of the required packages (for e.g. with conda)
```
$ conda create --name steam --file requirements.txt
```

Then download the dataset `games.csv` from this [link](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset) and put it inside the folder `data`

Then run the python file app.py and wait for a few seconds for the webapp to load and show all of the visualizations
```
$ python app.py
```
