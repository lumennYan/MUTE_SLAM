path = '../environment.yaml'
with open(path, 'r') as file:
    file_content = file.readlines()
with open(path, "w") as file:
    for line in file_content:
        for i, s in enumerate(line):
            if s == "=":
                line = line[:i]
        print(line, file=file)
