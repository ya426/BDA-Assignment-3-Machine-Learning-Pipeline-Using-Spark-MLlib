from pathlib import Path
from urllib.request import urlretrieve
p=Path(__file__).resolve().parent/'data'/'diabetes.csv'
p.parent.mkdir(exist_ok=True)
urlretrieve('https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv',p)
print('Downloaded:',p)
