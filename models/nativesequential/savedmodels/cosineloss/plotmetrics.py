import os
import pandas as pd

data = {'0': None}
for file in os.listdir('/home/aadidev/projects/player-cricket-detect/models/nativesequential/savedmodels/cosineloss'):
	with open(file) as file:
		logs = pd.read_csv(file)
		data[file] = logs

print(data)


# b10e10lr001training