import pandas as pd


domino_df = pd.read_csv('dominos7.csv')

domino_list = domino_df['0'].tolist()

domino_list.remove('4-9')
domino_list.remove('8-11')
domino_list.remove('9-10')
print(len(domino_list))
domino_list
