from SubFunctions import *


for dataset in [1, 2]:
    # Load features and labels without feature extraction
    data = ReadDataset(dataset, exec=True).read_data()

    # Perform analysis on the loaded data for the current dataset
    tP = Analysis(data, dataset)
    tP.ComparativeAnalysis()
    tP.PerformanceAnalysis()
    tP.ROCAnalysis()

