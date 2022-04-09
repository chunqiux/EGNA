from Bio import SeqIO
from typing import List
import os


class HHblits:
    """
    Run HHblits program for protein sequences.
    """
    def __init__(self, hhblits_program_path: str, database_path: str, n_cpu: int=1):
        """
        :param hhblits_program_path: The filepath of HHblits program
        :param database_path: The filepath of the searching sequence database
        :param n_cpu: Number of cpu cores for computation
        """
        self.hhblits_program_path = hhblits_program_path
        self.database_path = database_path
        self.n_cpu = n_cpu

    def run_hhblits(self, fa_path: str, out_dir: str):
        """
        :param fa_path: The path of the fasta file
        :param out_dir: The path of the output directory
        """
        fa_path_list = split_fasta(fa_path)
        for fa in fa_path_list:
            fa_name = os.path.basename(fa)
            out_path = os.path.join(out_dir, fa_name)
            os.system(
                f"{self.hhblits_program_path} -i {fa}"
                f" -d {self.database_path}"
                f" -cpu {self.n_cpu} -n 4 -e 0.001"
                f" -o {out_path}.hhr -ohhm {out_path}.hhm -oa3m {out_path}.a3m"
            )

def split_fasta(fasta_path: str) -> List[str]:
    """
    Read fasta file which contains multiple sequences and
    split it to multiple fasta files which contain a single sequence.
    :param fasta_path: File path of the original fasta file
    :return: A list of fasta file path
    """
    fa_path_list = []
    fa_dir = os.path.dirname(fasta_path)
    new_fa_dir = os.path.join(fa_dir, "tmp_fa_dir")
    if not os.path.exists(new_fa_dir):
        os.mkdir(new_fa_dir)
    with open(fasta_path, 'r') as fasta:
        for record in SeqIO.parse(fasta, 'fasta'):
            record_name = record.name.replace(':', '_')
            single_fasta_path = os.path.join(new_fa_dir, record_name)
            with open(single_fasta_path, 'w') as sf:
                SeqIO.write(record, sf, "fasta")
            fa_path_list.append(single_fasta_path)
    return fa_path_list