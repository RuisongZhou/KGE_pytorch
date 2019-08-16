import os


def CheckPath(path, raise_error=True):
    '''
    Used to check the existence of the path. If the path doesn't
    exist, raise error if raise_error is True, or make the path.
    '''
    if not os.path.exists(path):
        if raise_error:
            print("ERROR : Path %s does not exist!" % path)
            exit(1)
        else:
            print("WARNING : Path %s does not exist!" % path)
            print("INFO : Creating path %s." % path)
            os.makedirs(path)
            print("INFO : Successfully making dir!")
    return

'''
Used to print arguments on the screen.
'''
def printArgs(args):
    print("="*20 + "Arguments" + "="*20)
    argsDict = vars(args)
    for arg, value in argsDict.items():
        print("==> {} : {}".format(arg, value))
    print("="*50)

def loadTriple(fileName):
    with open(fileName, 'r') as fr:
        i = 0
        tripleList = []
        tripleDict = {}
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            tri = (head, rel, tail)
            tripleList.append(tri)
            tripleDict[tri] = True

    return tripleList, tripleDict
