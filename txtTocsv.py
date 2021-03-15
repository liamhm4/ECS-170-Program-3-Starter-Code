import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) < 3:
    print("python readfrom.txt writeto.csv [-c] [-p image.png]")
    quit()

hasColumnName = "-c" in sys.argv
wantsPlot = "-p" in sys.argv

plotFile = ""

if wantsPlot == True:
    hasColumnName = False
    plotFile = [x for x in sys.argv if x.endswith(".png")]
    if len(plotFile) == 0:
        print("python readfrom.txt writeto.csv -p image.png")
        quit()

readFrom = sys.argv[1]
writeTo = sys.argv[2]
columnTable = []

if hasColumnName:
    f = open(readFrom, "r")
    columns = f.readline()
    columnX = [x for x in [columns.split(" ")]]
    for i in columnX[0]:
        columnTable.append(i.strip())

# print(columnTable)
readFile = pd.read_csv(readFrom, sep=' ')
if hasColumnName:
    readFile.columns = columnTable
readFile.to_csv(writeTo)

if wantsPlot:
    data = np.genfromtxt(writeTo, delimiter=",", names=["_", "x", "y"])
    plt.xlabel('Frames')
    plt.ylabel('Values')
    plt.plot(data["x"], data["y"])
    plt.savefig(plotFile[0])