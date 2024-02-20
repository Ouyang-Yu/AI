
with open("创作.py") as f:
    print(max(f.readlines(),key= lambda x:len(x)))


with open("创作.py") as f:
    print(max(f,key=lambda x:len(x)))

with open("创作.py") as f:
    print(max(f,key= len))
# the longest line in a file