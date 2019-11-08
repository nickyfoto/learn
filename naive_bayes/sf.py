import numpy as np 
import pandas as pd 

df = pd.read_csv('emails.csv') #read the CSV file
print(df.head(5))

print(df.shape)