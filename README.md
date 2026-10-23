## <div align="center"> TITLE</div>
## DESIGN AND DEVELOPMENT OF A DISTRIBUTED ATTENTIONAL MIXED LEARNING ENABLED PHYSICS-BASED DEEP CONVOLUTIONAL NEURAL NETWORK FOR BRAIN TUMOR DETECTION
<br>

## HARDWARE REQUIREMENTS⚙️
<br>
    OS-Windows 11
<br>
    RAM-16GB
<br>
    ROM-More than 100 GB
<br>
    CPU-1.7 Ghz
<br>

## SOFTWARE REQUIREMENTS💻

<b>Software name: </b> Python: Version: <b>3.7.6</b>([Download link](https://www.python.org/ftp/python/3.7.6/python-3.7.6-amd64.exe))<br>
<b>Software name: </b> Pycharm: Version: <b>2024.1.7</b>([Download link](https://www.jetbrains.com/pycharm/download/download-thanks.html?platform=windows&code=PCC))
<br>

## HOW TO RUN➿
    Step 1: Loading the project in PYCHARM
        1. Open pycharm
        2. Go to File, select Open browse the project from your drive and select it. So that the
        project will get loaded into the Pycharm.
        3. For the first time, Pycharm will take some time to load the settings.
        4. Please wait if any process is loading at the bottom of the screen.
        5. Check the Project Interpreter (File -&gt; Settings -&gt; Project Interpreter).
        If this location “(C:\Users\---\AppData\Local\Programs\Python\Python37\python.exe) is
        not presented, then add this ‘python.exe’ from the installed location.

    Step 2: Generate the graphs plotted in the paper
        1. current project window in pycharm, Open ‘Main.py’, and click
        run button.
        2. When you execute this it will ask you to select anyone from two options like Yes (“Full
        analysis +Plots”) and NO (“Plots from pre evaluated data”).
        3. · If you select the first option, it will execute the complete analysis again and
        display the results. It may take time depending on the analysis.

        4. · Second option will display the results from the stored data i.e, mainly to skip the
        execution time.
        [Execution time expected: “Full analysis +Plots” — 48 hr]
        [Execution time expected: “Plots from pre evaluated data” – 30 sec]

## <div align="center">DESCRIPTION</div>

    1: preprocessing and Feature Extraction function (SRC-GetData.py : Line NO: 14-208)
        1.1. Preprocessing - (SRC-GetPreprocessing.py : Line NO: 8-50)
        1.2. Segmentation - (SRC-SetSegmentation.py : Line NO: 193-303)
        1.3. Distributed attentional fused pretrained model - (SRC-GetFeatures.py : Line NO: 51-73)
        1.4. GLCM features - (SRC-GetFeatures.py : Line NO: 26-47)
        1.5. Deep fused structural pattern - (SRC-GetFeatures.py : Line NO: 76-88)


    2: Training Percentage Analysis Part (SRC-Analysis.py : Line NO: 149-336)
        2.1. Comparative Analysis - (SRC-Analysis.py : Line NO: 164-219)
        2.2. Performance Analysis - (SRC-Analysis.py : Line NO: 222-273)
        2.3. ROC Analysis - (SRC-Analysis.py : Line NO: 275-336)


    3: Distributed attentional mixed learning enabled physics based DCNN (SRC-Model.py : Line NO: 402-505)
        3.1. DCNN(SRC-Model.py : Line NO: 422-485)
        3.2. Distributed Attention (SRC-Attention.py : Line NO: 454-460)
        3.3. Physics Loss (SRC-Model.py : Line NO: 388-399)


    4: Exixting Methods (SRC-Model.py : Line NO: 55-385)


See below for a installation for full documentation on training, testing.

<details open>
<summary>Packages Installation</summary>

```bash
pip install -r requirements.txt
```
</details>



## <div align="center">DATASET</div>

<i>Dataset1<b>([Click to Download te Dataset](Dataset Link: https://www.med.upenn.edu/cbica/brats2020/data.html))</b><br></i>
<br>
<i>Dataset2<b>([Click to Download te Dataset](Dataset Link: https://www.med.upenn.edu/sbia/brats2017/data.html))</b><br></i>
<br>
