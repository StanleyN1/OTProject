import csv

#Note that gprofiler needs to be installed to run this
from gprofiler import GProfiler


#Comment out all but one option from the different cases

mainCase = "PINK1"
#mainCase = "LRRK2"

#dim = 500
#dim = 1000
dim = 1500
#dim = 2000

#With LRRK2, there are no cases. Use the empty option.
#case = "_case1"
#case = "_case2"
#case = "_case3"
case = "_case4"
#case = ""


fileIn = mainCase + "_dim" + str(dim) + case + ".csv"

with open(fileIn) as geneFile:
    genesTemp = geneFile.readlines()
genesTemp = [line.strip() for line in genesTemp]
genesTemp = [line.split(',') for line in genesTemp]
genesTemp = genesTemp[1:len(genesTemp)]

#Only keep the first column
jt = 0
genes = ["" for row in genesTemp]
for gen in genesTemp:
    genes[jt] = gen[0]
    jt = jt + 1


# run gprofiler on the full list
gp = GProfiler(return_dataframe=True)
resBig = gp.profile(organism='hsapiens',query=genes,sources=['KEGG'],user_threshold=1,no_evidences = False)
outname = "gprofilerResults/outFull_" + mainCase + "_dim" + str(dim) + case + ".csv"
resBig.to_csv(outname,index=True)

print(str(len(genes)) + " genes in total")
