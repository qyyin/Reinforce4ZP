import sys 

def read_pros(fil):
    f = open(fil)
    d = {}
    while True:
        line = f.readline()
        if not line:break
        line = line.strip()
        if len(line) == 0 or line.startswith("#"):
            continue
        line = line.split("=")
        assert len(line) == 2
    
        name = line[0].strip()
        value = eval(line[1].strip())
        d[name] = value

    f.close()
    return d
