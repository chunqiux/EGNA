# EGNA
An empirical graph neural network for protein-ligand binding affinity prediction


## Preparation

### Download source code
At first, download the source code of EGNA from GitHub:

    $ git clone https://github.com/chunqiux/EGNA.git

### Install and activate Python3 environment
EGNA is implemented with Python3.7. We recommend you to use Anaconda to install the dependencies of
EGNA. Anaconda can be downloaded [here](https://www.anaconda.com/products/distribution).

After installing anaconda, create and activate the virtual environment as follows:

    $ conda env create -f egna_env.yml
    $ conda activate egna_env

When you want to quit the virtual environment, just:

    $ conda deactivate

### Install HHblits and sequence databases

HHblits is used to generate sequence profiles in our model. It can be downloaded
[here](https://github.com/soedinglab/hh-suite). The sequence databases can also be
downloaded in the same page. In this study, Uniclust30 is used.

## Usage
At first, prepare the PDB file of the protein structure and the sdf or mol2 file of
the ligand. It should be guaranteed that the two molecules are well docked or are extracted
from a true complex. Then, the pKd of the target complex can be predicted as follows:

    $ python predict.py -p protein.pdb -l ligand.mol2 -d "path of your sequene databases"

where '-p protein.pdb' means the path of protein file is 'protein.pdb', '-l ligand.mol2'
means the path of ligand file is 'ligand.mol2' and '-d' is used to designate the path of
sequence database for HHblits. Other options can be found by '-h'.

## License
<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.
