import os
print(f"Current Working Directory: {os.getcwd()}")
print(f"Files in this directory: {os.listdir('.')}")
if os.path.exists('data'):
    print(f"Files inside 'data' folder: {os.listdir('data')}")
else:
    print("The 'data' folder does not exist here!")