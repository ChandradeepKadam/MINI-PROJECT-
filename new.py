import os

print("Current Folder:", os.getcwd())
print("Files:", os.listdir())


df = pd.read_csv("car data.csv")