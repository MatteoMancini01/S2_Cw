## Acknowledgment
I would like to acknowledge the work of Thoeni, Andrew, Chris Budiselic, and Andrew in producing the replicated X-ray data of the Antikythera Mechanism in 2019. Their efforts in digitising and segmenting the fractured calendar ring sections have provided an invaluable foundation for this analysis.  
The dataset can be found in either my [GitHub Repository](https://github.com/MatteoMancini01/S2_Cw/tree/main) (see [data](https://github.com/MatteoMancini01/S2_Cw/tree/main/data)) or from the original source [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/VJGLVS)

I would like to acknowledge the invaluable assistance of ChatGPT-4 and Grammarly in the preparation of this project. The tools played a significant role in:

- Providing insights and suggestions for improving the structure and functionality of the code.
- Generating concise and accurate summaries of complex texts to enhance understanding and clarity.

While ChatGPT-4 contributed significantly to streamlining the development process and improving the quality of outputs, all results were rigorously reviewed, tested, and refined to ensure their accuracy, relevance, and alignment with project objectives.

## Introduction
In 1901 Captain Dimitrios Kontos and a crew of sponge divers, retrived numerous large objects from an ancient Roman cargo ship, 45 meters below the sea level, near the island of Antikythera (Greek island located in the Mediterranean Sea). Among the many objects retrived from the wreckage there was a tool, that is now know as the Antikythera mechanism.

The mechanism was designed by ancient Greek astronomers during the second century BC. This was the first known analog computer in human history. This consists of a ca;emdar ring with holes punched at the extrimity of its circumference. Unfortunaly approximatelly $25\%$ of the ring survived. We used to believe that the full ring contained $365$ holes, impling that the mechanism was used as a solar calendar. While, a new theory suggest that there were $354$ holes overall, i.e. the mechanism was a lunar calendar.

### Project Objective

In this project we aim to use an X-ray image of the calendar ring to then infer on the number of holes present in the comple ring, through Bayesian inference with Hamiltonian Monte Carlo. 

### Navigating the Repository

In this repository, you can find the following directories:
-	src, containing all the required Python scripts, in particular `plotting.py`, `original_model.py` and `model.py`, each containing may functions that serve for different purposes. Please check these files as they have a well structured and detailed docstring documentation.
-	csv_files, here we saved experiment outcomes as .csv files.
-	data, this directory contains few images of the Antikythera mechanism, with a ReadMe.txt and the data itself, saved as another .csv file.
-	plots, this directory contains all the saved outcomes displayed as plots.
-	relevant_material, here we collected the two main papers that performed the analysis before us.

And the following files:

-	AntikytheraMechanism_CourseworkProject.pdf, this pdf file contains all the instructions that we followed for this project.
-	NoteBook.ipynb, this is a Jupyter Notebook containing all the codes and information regarding this project.
-	NoteBook.py, a copy of NoteBook.ipynb converted into Python script
-	Report.pdf, this contains the report I wrote to summarise all the work done for this project.
-	requirements.txt, contains all the Python packages, one should install all the packages listed to run codes and the notebook.



## Getting started 
1. Create a Virtual Environment

   Run the following command to create your virtual environment

   ``` bash
    python -m venv <your_env>

- If the above command fails, please try:
   ```bash
   python3 -m venv <your_env>

Replace `<your_env>` with your preferred environment name, e.g. `stats_venv`.

2. Activate your virtual environment

  Activate your virtual environment with:
   ```bash
    source <your_env>/bin/activate
   ```
  Deactivate your environment with:
   ```bash
    deactivate
   ```
3. To install all the required libraries please run the command:
   ```bash
   pip install -r requirements.txt
   ```