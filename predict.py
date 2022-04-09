import argparse
from torch.utils.data import DataLoader
import shutil

from raw_features import RawFeatureExtraction, TMP_DATA_DIR
from data import *
from regression_model import EGNA


@torch.no_grad()
def predict(model_path, raw_feat):
    """
    Predict the pKd of a complex by using EGNA
    :param model_path: The path of the trained model of EGNA
    :param raw_feat: The generated raw features
    :return: The predicted pKd
    """
    raw_data = BindingData(raw_feat.res_feature_path, raw_feat.ligand_info_path, raw_feat.dist_info_path)
    dataloader = DataLoader(raw_data, 1, collate_fn=data_collate, pin_memory=True)

    net = load_model(model_path)
    net.eval()
    for data in dataloader:
        for j in range(2, 5):
            data[j] = data[j].to_sparse()
        pred = net(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7])

    return pred


def extract_raw_features(raw_feat: RawFeatureExtraction):
    """
    Featurize proteins, ligands and complexes
    :param raw_feat: The input RawFeatureExtraction class
    """
    print("Extracting protein sequences from the PDB file...\n")
    raw_feat.extract_sequences()
    print("Getting coordinates of heavy atoms in the protein...\n")
    raw_feat.get_protein_ha_coordinates()
    print("Extracting ligand raw features...\n")
    raw_feat.get_ligand_info()
    print("Computing distance matrices of the complex...\n")
    raw_feat.get_distance_matrix()
    print("Running HHblits and extracting protein raw features...\n")
    raw_feat.get_residue_features()


def load_model(model_path: str) -> EGNA:
    """
    Load trained EGNA models
    :param model_path: The path of the model file
    :return: The trained network
    """
    net = EGNA(30, 15)
    net.load_state_dict(torch.load(model_path))
    return net


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--protein", type=str, help="The PDB file of the protein")
    parser.add_argument("-l", "--ligand", type=str, help="The sdf or mol2 file of the ligand")
    parser.add_argument("-f", "--format", type=str, default="mol2", help='The format of the ligand file')
    parser.add_argument("--hhblits", type=str, default="hhblits", help='The path of hhblits program')
    parser.add_argument("-d", "--database", type=str, help="The path of the sequence database for hhblits")
    parser.add_argument("-u", "--cpu", type=int, default=4, help="The number of cpu cores used for hhblits")
    parser.add_argument("-o" "--output", type=str, default="", help="The path of the output file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_argument()

    rf = RawFeatureExtraction(args.hhblits, args.database, args.protein, args.ligand, args.format, args.cpu)
    extract_raw_features(rf)

    n_models = 5
    pkd_list = []
    for i in range(n_models):
        model_path = f"models/EGNA_{i}.pkl"
        pkd = predict(model_path, rf)
        pkd_list.append(pkd)

    mean_pred = sum(pkd_list) / n_models
    print(f"pKd of the input complex: {mean_pred[0]} Mol")

    if args.output != "":
        with open(args.output, "w") as of:
            of.write(mean_pred[0])

    shutil.rmtree(TMP_DATA_DIR)