#The code for DBCA is adapted from https://github.com/anmoisio/morphogen-dbca/blob/master/divide.py

from tqdm import tqdm
import torch
import random
import pandas as pd
from collections import defaultdict, Counter
from estnltk.vabamorf.morf import synthesize
from convert_data_to_unimorph import get_possible_forms_saldo, convert_unimorph_feats_to_saldo

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
#np.random.seed(SEED)

def get_lemma_feature_count(df_data, input_column_name):
    """Creates sets of functions/features, lemmas and their combinations in the DataFrame.

    Args:
        df_data (pd.DataFrame): Pandas DataFrame that includes a column containing the input sequences
        input_column_name (str): Column that contains the input sequences

    Returns:
        lemmas_set: Set of all lemmas 
        features_set: Set of all UniMorph features
        feature_combinations_set: Set of all feature combinations
        compounds_set: Set of all compounds (compound = features + lemma)
        total_count: Total count of all atoms (features + lemmas)
    """
    lemmas_set = set()
    features_set = set()
    feature_combinations_set = set()
    compounds_set = set()
    total_count = 0

    for i, row in df_data.iterrows():
        all_atoms = row[input_column_name].split()
        for atoms in all_atoms:
            total_count += 1
            compounds_set.add(atoms)
            features = atoms.split("(")[0]
            feature_combinations_set.add(features)
            features_set.update(features.split(";"))
            lemma = atoms.split("(")[1][:-1]
            lemmas_set.add(lemma)
            
    return lemmas_set, features_set, feature_combinations_set, compounds_set, total_count


def get_weight_feature_set(feature_set, counts_counter):
    """Calculates the complement of the probability that feature_set occurs with a specific lemma.

    Args:
        feature_set (str): Set of features to get the weight for (str because joined using ;)
        counts_counter (Counter): Counter that contains the number of occurrences of feature_set for each lemma

    Returns:
        weight (float): complement of the probability that feature_set occurs with the lemma that it most often occurs with
    """
    total_counts = sum(counts_counter[feature_set].values())
    max_counts = counts_counter[feature_set].most_common(1)[0][1]
    weight = 1 - (max_counts/total_counts)

    return weight


def filter_data(df_data, input_column_name, output_column_name, original_words_column_name, original_text_column, threshold_weight=0.25):
    """Filters out all compounds where the feature combination has a weight lower than the threshold.

    Args:
        df_data (pd.DataFrame): DataFrame that contains both the input and output sequences.
        input_column_name (str): Column name of the column that contains the input sequences.
        output_column_name (str): Column name of the column that contains the output sequences.
        original_words_column_name (str): Column name of the column that contains the original word forms.
        original_text_column (str): Column name of the column that contains the original text.
        threshold_weight (float): Compounds where the feature combination has a weight lower than this will be filtered out.

    Returns:
        df_new (pd.DataFrame): Filtered DataFrame
    """
    cnt_tag_combinations = defaultdict(Counter)
    tag_combinations_to_remove = set()

    for i, row in df_data.iterrows():
        all_atoms = row[input_column_name].split()
        for atoms in all_atoms:
            features = atoms.split("(")[0]
            lemma = atoms.split("(")[1][:-1]
            cnt_tag_combinations[features][lemma] += 1

    for key in cnt_tag_combinations.keys():
        weight = get_weight_feature_set(key, cnt_tag_combinations)
        if weight < threshold_weight:
            tag_combinations_to_remove.add(key)
    print(f"Removing the following feature combinations with weight lower than {threshold_weight}: {tag_combinations_to_remove}")
    
    original_texts = []
    input_strs = []
    output_strs = []
    original_strs = []
    for i, row in df_data.iterrows():
        in_items, out_items, original_items, original_text = row[input_column_name].split(), row[output_column_name].split(), row[original_words_column_name].split(), row[original_text_column]
        in_items_filtered, out_items_filtered, original_items_filtered = [], [], []
        for in_item, out_item, original_item in zip(in_items, out_items, original_items):
            tag_combination = in_item.split("(")[0]
            if tag_combination not in tag_combinations_to_remove:
                in_items_filtered.append(in_item)
                out_items_filtered.append(out_item)
                original_items_filtered.append(original_item)
        if in_items_filtered and out_items_filtered:
            original_texts.append(original_text)
            input_strs.append(" ".join(in_items_filtered))
            output_strs.append(" ".join(out_items_filtered))
            original_strs.append(" ".join(original_items_filtered))
                
    df_new = pd.DataFrame()
    df_new[original_text_column] = original_texts
    df_new[original_words_column_name] = original_strs
    df_new[input_column_name] = input_strs
    df_new[output_column_name] = output_strs
    
    return df_new


def get_data_matrices_lexical(df_data, input_column_name):
    """Turns the data into matrices of counts for both atoms and compounds where atom is either a feature or a lemma and
    a compound consists of a feature set and a lemma.

    Args:
        df_data (pd.DataFrame): DataFrame that contains a column with the input sequences.
        input_column_name (str): Column name of the column that contains the input sequences.

    Returns:
        atom_freq_matrix (torch.tensor): N x A matrix where N is the number of rows in the DataFrame and A is the number of atoms
        com_freq_matrix (torch.tensor): N x C matrix where N is the number of rows in the DataFrame and C is the number of compounds
        atom_ids (dict): Dictionary where the keys are atoms and values their ids
        compound_ids (dict): Dictionary where the keys are compounds and values their ids
    """
    lemmas_set, features_set, _, compounds_set, _ = get_lemma_feature_count(df_data, input_column_name)
    compound_ids = {k: i for i, k in enumerate(compounds_set)}
    atoms = lemmas_set.union(features_set)
    atom_ids = {k: i for i, k in enumerate(atoms)}
    num_atoms = len(atoms)
    num_compounds = len(compounds_set)

    atom_freq_matrix = torch.zeros((0, num_atoms), dtype=torch.uint8).to_sparse()
    com_freq_matrix = torch.zeros((0, num_compounds), dtype=torch.uint8).to_sparse()
    
    for i, row in df_data.iterrows():
        a_row = torch.zeros(num_atoms, dtype=torch.uint8)
        c_row = torch.zeros(num_compounds, dtype=torch.uint8)
        compounds = row[input_column_name].split()
        for compound in compounds:
            lemma = compound.split("(")[1][:-1]
            features = compound.split("(")[0].split(";")
            for atom in [lemma] + features:
                a_row[atom_ids[atom]] += 1
            c_row[compound_ids[compound]] += 1
                        
        if torch.sum(a_row) > 0:
            atom_freq_matrix = torch.cat((atom_freq_matrix, a_row.unsqueeze(0).to_sparse()), dim=0)
            com_freq_matrix = torch.cat((com_freq_matrix, c_row.unsqueeze(0).to_sparse()), dim=0)

    print(f"Number of different atoms: {atom_freq_matrix.shape[1]}")
    print(f"Total number of atoms: {atom_freq_matrix.sum()}")
    print(f"Number of different compounds: {com_freq_matrix.shape[1]}")
    print(f"Total number of compounds: {com_freq_matrix.sum()}")
    return atom_freq_matrix, com_freq_matrix, atom_ids, compound_ids

def get_data_matrices_structural(df_data, input_column_name):
    """Turns the data into matrices of counts for both atoms and compounds where atom is a feature and a compound is a feature set.

    Args:
        df_data (pd.DataFrame): DataFrame that contains a column with the input sequences.
        input_column_name (str): Column name of the column that contains the input sequences.

    Returns:
        atom_freq_matrix (torch.tensor): N x A matrix where N is the number of rows in the DataFrame and A is the number of atoms
        com_freq_matrix (torch.tensor): N x C matrix where N is the number of rows in the DataFrame and C is the number of compounds
        atom_ids (dict): Dictionary where the keys are atoms and values their ids
        compound_ids (dict): Dictionary where the keys are compounds and values their ids
    """
    _, features_set, compounds_set, _, _ = get_lemma_feature_count(df_data, input_column_name)
    compound_ids = {k: i for i, k in enumerate(compounds_set)}
    atoms = features_set
    atom_ids = {k: i for i, k in enumerate(atoms)}
    num_atoms = len(atoms)
    num_compounds = len(compounds_set)

    atom_freq_matrix = torch.zeros((0, num_atoms), dtype=torch.uint8).to_sparse()
    com_freq_matrix = torch.zeros((0, num_compounds), dtype=torch.uint8).to_sparse()
    
    for i, row in df_data.iterrows():
        a_row = torch.zeros(num_atoms, dtype=torch.uint8)
        c_row = torch.zeros(num_compounds, dtype=torch.uint8)
        compounds = row[input_column_name].split()
        for compound_lemma in compounds:
            compound = compound_lemma.split("(")[0]
            features = compound.split(";")
            for atom in features:
                a_row[atom_ids[atom]] += 1
            c_row[compound_ids[compound]] += 1
                        
        if torch.sum(a_row) > 0:
            atom_freq_matrix = torch.cat((atom_freq_matrix, a_row.unsqueeze(0).to_sparse()), dim=0)
            com_freq_matrix = torch.cat((com_freq_matrix, c_row.unsqueeze(0).to_sparse()), dim=0)

    print(f"Number of different atoms: {atom_freq_matrix.shape[1]}")
    print(f"Total number of atoms: {atom_freq_matrix.sum()}")
    print(f"Number of different compounds: {com_freq_matrix.shape[1]}")
    print(f"Total number of compounds: {com_freq_matrix.sum()}")
    return atom_freq_matrix, com_freq_matrix, atom_ids, compound_ids

##############################################DBCA#####################################################################
# Code for DBCA
# Adapted from: https://github.com/anmoisio/morphogen-dbca/blob/master/divide.py
def chernoff_coef(matrix1, matrix2, alpha):
    """
    The Chernoff coefficient c is a similarity measure C_{alpha}(P||Q)
    = sum_k[p_k^alpha * q_k^(1-alpha)] e[0,1] between two probability 
    distributions P and Q. The alpha parameter determines if we want to
    measure whether Q includes elements that are not in P.
    (Atom divergence is 1 - Chernoff coefficient, with alpha=0.5)
    (Compound divergence is 1 - Chernoff coefficient, with alpha=0.1)

    Returns
    -------
    torch.Tensor, vector with length matrix.shape[0]
        Chernoff coefficient between vector and each row of matrix
    """
    if len(matrix1.shape) == 1:
        sum_axis = 0
    elif len(matrix1.shape) == 2:
        sum_axis = 1
        if len(matrix2.shape) == 1:
            if matrix2.shape[0] != matrix1.shape[1]:
                raise ValueError("matrix second dim must be equal to vector length")
    else:
        raise ValueError("matrix must be 1D or 2D")
    if alpha < 0 or alpha > 1:
        raise ValueError("alpha must be in [0,1]")

    return torch.sum(torch.exp((torch.log(matrix1) * alpha) +
                               (torch.log(matrix2) * (1-alpha))), axis=sum_axis)

def mat_vec_divergence(matrix, vector, alpha):
    """Divergence between a vector and each row of a matrix."""
    return 1 - chernoff_coef(normalize_matrix(matrix), normalize_vector(vector), alpha)

def vec_vec_divergence(vector1, vector2, alpha):
    """Divergence between two vectors."""
    return 1 - torch.clamp(chernoff_coef(normalize_vector(vector1), normalize_vector(vector2), alpha), min=0.0, max=1.0)

def get_scores(atomdivs, comdivs, target_atom_div, target_com_div):
    """Return the scores for each index in atomdivs and comdivs vectors."""
    return - torch.abs(comdivs - target_com_div) \
           - torch.abs(atomdivs - target_atom_div)

def normalize_vector(vector):
    """Normalize a vector to have sum 1."""
    return torch.nan_to_num(torch.divide(vector, torch.sum(vector)))

def normalize_matrix(matrix):
    """Normalize a matrix; each row sums up to 1."""
    return torch.nan_to_num(
        torch.transpose(
            torch.divide(
                torch.transpose(matrix, 0, 1),
                torch.sum(matrix, axis=1)
            ),
        0, 1) # /transpose
    )

def get_subset_indices(subset_indices_tensor, subset_id):
    """Returns the indices of the given subset."""
    return (subset_indices_tensor == subset_id).nonzero()

def get_random_subsample(subset_indices_tensor, subset_id, subsample_size, atom_freq_matrix, com_freq_matrix, device):
    """Take a random subsample of the sentences in the given subset."""
    subset_indices = get_subset_indices(subset_indices_tensor, subset_id).squeeze()
    if subset_indices.shape:
        random_subsample_indices = subset_indices[
        torch.randperm(subset_indices.shape[0], device=device)][:subsample_size]
    else:
        random_subsample_indices = subset_indices.unsqueeze(0)
    random_atom = atom_freq_matrix.index_select(0, random_subsample_indices).to_dense()
    random_com = com_freq_matrix.index_select(0, random_subsample_indices).to_dense()
    return random_subsample_indices, random_atom, random_com

def subsample(subset_indices_tensor, subset_com_freq_sum, subset_atom_freq_sum, from_set, to_sets, subsample_size, 
              atom_freq_matrix, com_freq_matrix, candidate_com_sums, candidate_atom_sums, device):
    """Take a random subsample of sentences in from_set, of size self.subsample_size.
    Update candidate_com_sums and candidate_atom_sums with the new subsample."""
    random_idxs, random_atom, random_com = get_random_subsample(subset_indices_tensor, from_set, subsample_size, atom_freq_matrix,
                                                               com_freq_matrix, device)

    #used_ids_mask = torch.zeros(subsample_size, device=device)
    for to_set in to_sets:
        # a new matrix that has the freq sums with each new sample and the new random subset
        candidate_com_sums[to_set] = subset_com_freq_sum[to_set] + random_com
        candidate_atom_sums[to_set] = subset_atom_freq_sum[to_set] + random_atom
    return random_idxs, candidate_com_sums, candidate_atom_sums

def update_sets(subset_indices, subset, selected_idx, subset_com_freq_sum, subset_atom_freq_sum, 
                sampled_idx, com_freq_matrix, atom_freq_matrix):
    """Update the data structs after a new sample has been selected to a subset."""

    
    # mark the selected index as used
    #used_ids_mask[selected_idx] = -torch.inf

    # map the selected index to the original
    selected_idx = sampled_idx[selected_idx]

    subset_indices[selected_idx] = subset
    selected_com_row = com_freq_matrix[selected_idx].to_dense()
    selected_atom_row = atom_freq_matrix[selected_idx].to_dense()
    subset_com_freq_sum[subset] += selected_com_row
    subset_atom_freq_sum[subset] += selected_atom_row
    return subset_indices, subset_com_freq_sum, subset_atom_freq_sum

def candidate_divergences(subset_id, subset_atom_freq_sum, subset_com_freq_sum, candidate_atom_sums, candidate_com_sums,
                         train_set_id=0, test_set_id=1):
    """Compute the compound divergence between the subset vector and the matrix containing
    the candidate sentences."""
    if subset_id == train_set_id:
        the_other_set_id = test_set_id
        com_alpha = 0.1
    else:
        the_other_set_id = train_set_id
        com_alpha = 0.9 # because test set is the first argument to chernoff_coef()
    # The atom divergence is 1 - Chernoff coefficient, with alpha=0.5.
    atom_div_add_to_subset = mat_vec_divergence(candidate_atom_sums[subset_id],
        subset_atom_freq_sum[the_other_set_id], 0.5)
    # The compound divergence is 1 - Chernoff coefficient, with alpha=0.1.
    com_div_add_to_subset = mat_vec_divergence(candidate_com_sums[subset_id],
        subset_com_freq_sum[the_other_set_id], com_alpha)
    return atom_div_add_to_subset, com_div_add_to_subset

def best_sentence(atom_div_vec, com_div_vec, target_atom_div, target_com_div, mask=None):
    """Return the argmax and max score, and the compound and atom divergences of
    the sample that maximises the score. Score is the linear combination of
    the negated differences between the target divergences and actual divergences.
    Optionally, a mask can be applied to the score vector, e.g. to avoid selecting
    sentences that have already been selected."""
    scores = get_scores(atom_div_vec, com_div_vec, target_atom_div, target_com_div)
    if mask is not None:
        scores += mask
    best_idx = torch.argmax(scores)
    return {'idx': best_idx,
            'score': scores[best_idx],
            'atomdiv': atom_div_vec[best_idx],
            'comdiv': com_div_vec[best_idx]}

def get_train_test_split_dbca(full_atom_matrix, full_compound_matrix, target_com_div=1.0, min_test_percent=0.05,
                          max_test_percent=0.3, subsample_size = 100, device="cpu"):
    """Assigns split (train/test) indices to each row of the data, aiming to minimize atom divergence while maximizing compound divergence.

    Args:
        full_atom_matrix (torch.tensor): Sparse matrix containing the counts of each atom in each data row
        full_compound_matrix (torch.tensor): Sparse matrix containing the counts of each compound in each data row
        target_com_div (float): Target value for the compound divergence
        min_test_percent (float): Minimum share of rows to assign to the test split
        max_test_percent (float): Maximum share of rows to assign to the test split
        subsample_size (int): The number of rows to sample for selecting the next train/test item
        device (str): Device to use

    Returns:
        subset_indices (torch.tensor): 1D Tensor containing split (train/test) indices for the data rows.
    """
    
    train_set_id = 0
    test_set_id = 1
    discard_set_id = 2
    target_atom_div = 0.0
    
    atom_freq_matrix = full_atom_matrix.to(device)
    com_freq_matrix = full_compound_matrix.to(device)
    n_samples = full_atom_matrix.shape[0]

    # Get number of atoms and compounds
    com_dim = com_freq_matrix.shape[1]
    atom_dim = atom_freq_matrix.shape[1]

    # keep unnormalised vectors as separate variables to enable updating
    subset_com_freq_sum = torch.zeros((2, com_dim), device=device)
    subset_atom_freq_sum = torch.zeros((2, atom_dim), device=device)

    # init subset indices in the discard set
    subset_indices = torch.zeros(n_samples, device=device) + discard_set_id

    best_values = {}
    train_size = 0
    test_size = 0

    random_train_example_id = random.randrange(n_samples)
    #used_ids_mask = torch.tensor([0.0])
    sampled_idx = 0

    subset_indices, subset_com_freq_sum, subset_atom_freq_sum = update_sets(subset_indices, 
                                                                            train_set_id, 
                                                                            0, 
                                                                            subset_com_freq_sum, 
                                                                            subset_atom_freq_sum,
                                                                            torch.tensor(random_train_example_id).unsqueeze(0),
                                                                            com_freq_matrix,
                                                                            atom_freq_matrix)

    print('Initialised the train set with one random sentence.')
    
    print('Starting division. ' + \
    f'Train set size: {get_subset_indices(subset_indices, train_set_id).size()[0]}, ' + \
    f'Test set size: {get_subset_indices(subset_indices, test_set_id).size()[0]}.')
    print(f'Dividing {n_samples} sentences...')
    for i in tqdm(range(n_samples-1)):
        train_size = get_subset_indices(subset_indices, train_set_id).size()[0]
        test_size  = get_subset_indices(subset_indices, test_set_id).size()[0]
        test_percent = test_size / (train_size + test_size)
        subsample_size = min(subsample_size, get_subset_indices(subset_indices, discard_set_id).shape[0])
        candidate_com_sums = torch.zeros((2, subsample_size, com_dim), device=device)
        candidate_atom_sums = torch.zeros((2, subsample_size, atom_dim),device=device)
        random_idx, candidate_com_sums, candidate_atom_sums = subsample(subset_indices, 
                                                                        subset_com_freq_sum, 
                                                                        subset_atom_freq_sum, 
                                                                        discard_set_id, 
                                                                        [train_set_id, test_set_id],
                                                                        subsample_size,
                                                                        atom_freq_matrix, 
                                                                        com_freq_matrix,
                                                                        candidate_com_sums,
                                                                        candidate_atom_sums,
                                                                        device)
        
        
        if test_percent > max_test_percent: # First check the size constraints
            atom_div_add_to_subset, com_div_add_to_subset = candidate_divergences(train_set_id, 
                                                                                  subset_atom_freq_sum, 
                                                                                  subset_com_freq_sum, 
                                                                                  candidate_atom_sums, 
                                                                                  candidate_com_sums)
            
            best_values[train_set_id] = best_sentence(atom_div_add_to_subset, 
                                                   com_div_add_to_subset, 
                                                   target_atom_div, 
                                                   target_com_div)
            
            selected_set = train_set_id
        elif test_percent < min_test_percent:
            atom_div_add_to_subset, com_div_add_to_subset = candidate_divergences(test_set_id, 
                                                                                  subset_atom_freq_sum, 
                                                                                  subset_com_freq_sum, 
                                                                                  candidate_atom_sums, 
                                                                                  candidate_com_sums)
            
            best_values[test_set_id] = best_sentence(atom_div_add_to_subset, 
                                                       com_div_add_to_subset, 
                                                       target_atom_div, 
                                                       target_com_div)
            selected_set = test_set_id
        else: # otherwise compare the max scores of the two sets
            atom_div_add_to_subset, com_div_add_to_subset = candidate_divergences(train_set_id, 
                                                                                  subset_atom_freq_sum, 
                                                                                  subset_com_freq_sum, 
                                                                                  candidate_atom_sums, 
                                                                                  candidate_com_sums)
            
            best_values[train_set_id] = best_sentence(atom_div_add_to_subset, 
                                                       com_div_add_to_subset, 
                                                       target_atom_div, 
                                                       target_com_div)

            atom_div_add_to_subset, com_div_add_to_subset = candidate_divergences(test_set_id, 
                                                                                  subset_atom_freq_sum, 
                                                                                  subset_com_freq_sum, 
                                                                                  candidate_atom_sums, 
                                                                                  candidate_com_sums)
            
            best_values[test_set_id] = best_sentence(atom_div_add_to_subset, 
                                                       com_div_add_to_subset, 
                                                       target_atom_div, 
                                                       target_com_div)
                
                
            if best_values[train_set_id]['score'] > best_values[test_set_id]['score']:
                selected_set = train_set_id
            else:
                selected_set = test_set_id

        
        selected_idx = best_values[selected_set]['idx']
        subset_indices, subset_com_freq_sum, subset_atom_freq_sum = update_sets(subset_indices, 
                                                                                selected_set, 
                                                                                selected_idx, 
                                                                                subset_com_freq_sum, 
                                                                                subset_atom_freq_sum,
                                                                                random_idx,
                                                                                com_freq_matrix,
                                                                                atom_freq_matrix)
    
    
    return subset_indices


def get_atom_compound_div(atom_freq_full_matrix, com_freq_full_matrix, split_indices, train_set_id = 0, test_set_id = 1, 
                          alpha_atom = 0.5, alpha_compound=0.1):
    """Calculates the atom and compound divergence for two data splits.

    Args:
        atom_freq_full_matrix (torch.tensor): Sparse matrix containing the counts of each atom in each data row
        com_freq_full_matrix (torch.tensor): Sparse matrix containing the counts of each compound in each data row
        split_indices (torch.tensor): 1D Tensor containing split (train/test) indices for the data rows.
        train_set_id (int): Train set ID in split_indices
        test_set_id (int): Test set ID in split_indices
        alpha_atom (float): Alpha for calculating the Chernoff coefficient for atoms
        alpha_compund (float): Alpha for calculating the Chernoff coefficient for compounds
        

    Returns:
        atom_div (float): Atom divergence value between the two data splits
        com_div (float): Compound divergence value between the two data splits
    """

    train_indices = (split_indices == train_set_id).nonzero().squeeze()
    test_indices = (split_indices == test_set_id).nonzero().squeeze()
    train_atom_freq_sums = atom_freq_full_matrix.index_select(0, train_indices).sum(0).to_dense()
    test_atom_freq_sums = atom_freq_full_matrix.index_select(0, test_indices).sum(0).to_dense()
    train_com_freq_sums = com_freq_full_matrix.index_select(0, train_indices).sum(0).to_dense()
    test_com_freq_sums = com_freq_full_matrix.index_select(0, test_indices).sum(0).to_dense()
    
    atom_div = vec_vec_divergence(train_atom_freq_sums, test_atom_freq_sums, alpha_atom)
    com_div = vec_vec_divergence(train_com_freq_sums, test_com_freq_sums, alpha_compound)
    return atom_div.item(), com_div.item()

#############################################DBCA#################################################################

#Systematicity
def get_train_test_split_systematicity(atom_matrix_sparse, compound_matrix_sparse, atom_combinations_to_test, 
                                       atom_ids_dict, compound_ids_dict, train_set_id = 0, test_set_id=1):
    
    """Assigns split (train/test) indices to each row of the data, where rows containing combinations of atoms in atom_combinations_to_test
    are assigned to the test split.

    Args:
        atom_matrix_sparse (torch.tensor): Sparse matrix containing the counts of each atom in each data row
        compound_matrix_sparse (torch.tensor): Sparse matrix containing the counts of each atom in each data row
        atom_combinations_to_test (List[List]): List of lists containing the atom combinations to assign to the test set
        atom_ids_dict (dict): Dictionary with atom values as keys and ids as values
        compound_ids_dict (dict): Dictionary with compound values as keys and ids as values
        train_set_id (int): Split id to assign to the train set items
        test_set_id (int): Split id to assign to the test set items
        

    Returns:
        subset_indices (torch.tensor): 1D Tensor containing split (train/test) indices for the data rows.
    """
    atom_matrix = atom_matrix_sparse.to_dense()
    compound_matrix = compound_matrix_sparse.to_dense()
    n_samples = compound_matrix.shape[0]
    subset_indices = torch.zeros(n_samples)
    subset_indices[:] = train_set_id
    test_indices = set()
    for atoms in atom_combinations_to_test:
        compound_ids_to_test = [v for k, v in compound_ids_dict.items() if 
                                set(atoms).issubset(k.replace("(", ";").replace(")", "").split(";"))]
        mask = (compound_matrix[:, compound_ids_to_test] > 0).any(dim=1)
        test_indices.update(torch.where(mask)[0])  
    subset_indices[list(test_indices)] = test_set_id

    #Check if train split includes all atoms
    train_atom_counts = atom_matrix.index_select(0, get_subset_indices(subset_indices, train_set_id).squeeze()).sum(0)
    for atoms in atom_combinations_to_test:
        atom_ids_to_check = [atom_ids_dict[atom] for atom in atoms]
        all_present = (train_atom_counts[atom_ids_to_check] > 0).all().item()
        if all_present:
            print(f"All atoms in {atoms} present in the train set.")
        else:
            ids_to_atoms = {v : k for k, v in atom_ids_dict.items()}
            missing = torch.where(train_atom_counts[atom_ids_to_check] == 0)[0].tolist()
            print(f"WARNING! Not all atoms in {atoms} present in the train set. Missing atom(s): {[ids_to_atoms[atom_ids_to_check[i]] for i in missing]}")
        
    print("Train size:", get_subset_indices(subset_indices, train_set_id).squeeze().shape[0])
    print("Test size:", get_subset_indices(subset_indices, test_set_id).squeeze().shape[0])
    return subset_indices

#Productivity
def get_train_test_split_productivity(atom_matrix_sparse, compound_matrix_sparse, atom_ids_dict, compound_ids_dict, max_train_len = 5, 
                                      train_set_id=0, test_set_id=1, sentence_level = True):
    """Assigns split (train/test) indices to each row of the data, where rows with input length > max_train_len are assigned to the test set

    Args:
        atom_matrix_sparse (torch.tensor): Sparse matrix containing the counts of each atom in each data row
        compound_matrix_sparse (torch.tensor): Sparse matrix containing the counts of each compound in each data row
        atom_ids_dict (dict): Dictionary with atom values as keys and ids as values
        compound_ids_dict (dict): Dictionary with compound values as keys and ids as values
        max_train_len (int): maximum input length in the train set
        train_set_id (int): Split id to assign to the train set items
        test_set_id (int): Split id to assign to the test set items
        sentence_level (bool): length measured as number of compounds if True, as number of morphological features in a compound if False
        

    Returns:
        subset_indices (torch.tensor): 1D Tensor containing split (train/test) indices for the data rows.
    """
    
    compound_matrix = compound_matrix_sparse.to_dense()
    atom_matrix = atom_matrix_sparse.to_dense()
    n_samples = atom_matrix.shape[0]
    subset_indices = torch.zeros(n_samples) + train_set_id
    
    if sentence_level:
        test_indices = (compound_matrix.sum(dim=1) > max_train_len).nonzero().squeeze()
    else:
        test_compound_ids = []
        for comp, ind in compound_ids_dict.items():
            if len(comp.split(";")) > max_train_len:
                test_compound_ids.append(ind)
        mask = (compound_matrix[:, test_compound_ids] > 0).any(dim=1)
        test_indices = torch.where(mask)[0]

    subset_indices[test_indices] = test_set_id
    
    #Check that all atoms are present in the training set
    atom_matrix_train = atom_matrix.index_select(0, get_subset_indices(subset_indices, train_set_id).squeeze())
    missing_atom_indices = (atom_matrix_train.sum(dim=0) == 0).nonzero().squeeze().tolist()
    if missing_atom_indices:
        ids_to_atoms = {v : k for k, v in atom_ids_dict.items()}
        print(f"WARNING! Not all atoms present in the train set. Missing atom(s): {[ids_to_atoms[i] for i in missing_atom_indices]}")
        
    
    
    print("Train size:", get_subset_indices(subset_indices, train_set_id).squeeze().shape[0])
    print("Test size:", get_subset_indices(subset_indices, test_set_id).squeeze().shape[0])
    return subset_indices


#Substitutivity

def get_random_split(n_samples, test_ratio=0.2, train_set_id = 0, test_set_id=1):
    """Randomly assigns samples to either the train or test set.

    Args:
        n_samples (int): Number of samples to assign to train/test
        test_ratio (float): Share of samples to assign to the test set
        train_set_id (int): Split id to assign to the train set items
        test_set_id (int): Split id to assign to the test set items
        

    Returns:
        subset_indices (torch.tensor): 1D Tensor containing split (train/test) indices for the data rows.
    """

    num_test = int(n_samples * test_ratio)
    subset_indices = torch.zeros(n_samples) + train_set_id
    shuffled_indices = torch.randperm(n_samples)
    test_indices = shuffled_indices[:num_test]
    subset_indices[test_indices] = test_set_id
    
    return subset_indices

def get_train_test_df(df_data, subset_indices, train_set_id = 0, test_set_id = 1):
    """Splits a DataFrame containing all data into two separate DataFrames based on the indices in subset_indices

    Args:
        df_data (pd.DataFrame): DataFrame containing all of the data
        subset_indices (torch.tensor): 1D Tensor containing split (train/test) indices for the data rows.
        train_set_id (int): Split id to assign to the train set items
        test_set_id (int): Split id to assign to the test set items
        

    Returns:
        df_train (pd.DataFrame): DataFrame containing all rows from the original DataFrame assigned to the train set
        df_test (pd.DataFrame): DataFrame containing all rows from the original DataFrame assigned to the test set
    """

    split_ids = subset_indices.cpu().numpy()
    df_train = df_data[split_ids == train_set_id]
    df_test = df_data[split_ids == test_set_id]
    df_train.reset_index(inplace=True, drop=True)
    df_test.reset_index(inplace=True, drop=True)
    return df_train, df_test


#Currently only works for morphological features, does not replace lemmas
def replace_atoms_with_synonyms(input_seq, atom_to_syn_dict, p_replace=1.0):
    """Replaces all occurrences of atoms specified in atom_to_syn_dict with the "synonyms" specified in the same dict

    Args:
        input_seq (str): Input string consisting of a number of compounds
        atom_to_syn_dict (dict): Dictionary containing the atoms to replace as keys and their "synonyms" as values
        p_replace (float): Probability of replacing an atom with its synonym

    Returns:
        new_input_seq (str): The new input sequence where atoms have been replaced with their synonyms
    """

    all_compounds = input_seq.split()
    new_compounds = []
    for compound in all_compounds:
        features_list = compound.split("(")[0].split(";")
        lemma = compound.split("(")[1][:-1]
        if p_replace == 1:
            new_features_list = [atom_to_syn_dict.get(item, item) for item in features_list]
        else:
            new_features_list = []
            for item in features_list:
                if item in atom_to_syn_dict.keys() and random.choices([True, False], weights = [p_replace,1-p_replace], k=1)[0]:
                    new_features_list.append(atom_to_syn_dict[item])
                else:
                    new_features_list.append(item)
        lemma = atom_to_syn_dict.get(lemma, lemma)
        comp = ";".join(new_features_list)
        new_compounds.append(f"{comp}({lemma})")
    new_input_seq = " ".join(new_compounds)
    return new_input_seq


def get_atom_synonym_counts(df_data, input_column_name, atom_to_syn_dict):
    """Counts the total number of each atom and its synonym specified in atom_to_syn_dict

    Args:
        df_data (pd.DataFrame): DataFrame that contains both input and output sequences
        input_column_name (str): Name of the column in df_data that contains the input sequences
        atom_to_syn_dict (dict): Dictionary containing the atoms to replace as keys and their "synonyms" as values

    Returns:
        counts_dict (dict): Dictionary containing the total count of each atom and its synonym
    """

    counts_dict = dict()
    my_column = df_data[input_column_name].str.replace("(", ";").str.replace(")", ";").str.replace(" ", "").str.split(";")
    for atom, syn in atom_to_syn_dict.items():
        count_atom = my_column.apply(lambda lst: lst.count(atom)).sum().item()
        count_syn = my_column.apply(lambda lst: lst.count(syn)).sum().item()
        counts_dict[(atom, count_atom)] = (syn, count_syn)

    return counts_dict


def prepare_substitutivity_test(df_data, input_column_name, subset_indices, atom_to_syn_dict,
                                train_set_id = 0, test_set_id = 1, p_replace_train=0.5):
    """Splits the full DataFrame with data into a train and a test split according to the subset indices, followed by creating a duplicate 
        of the test set where all atoms specified in atom_to_syn_dict are replaced with their synonyms and thereafter replacing atoms with
        their synonyms in the train set with probability p_replace_train.

    Args:
        df_data (pd.DataFrame): DataFrame that contains both input and output sequences
        input_column_name (str): Name of the column in df_data that contains the input sequences
        subset_indices (torch.tensor): 1D tensor containing the split (train/test) indices for each row in df_data
        atom_to_syn_dict (dict): Dictionary containing the atoms to replace as keys and their "synonyms" as values
        train_set_id (int): Split id for the items to be assigned to the train set
        test_set_id (int): Split id for the items to be assigned to the train set
        p_replace_train (float): Probability with which to replace atoms with their synonyms in the train set

    Returns:
        df_train (pd.DataFrame):
        df_test_orig (pd.DataFrame):
        df_test_syn (pd.DataFrame):
    """
    train_indices = get_subset_indices(subset_indices, train_set_id).squeeze()
    test_indices = get_subset_indices(subset_indices, test_set_id).squeeze()
    df_train, df_test_orig = get_train_test_df(df_data, subset_indices, train_set_id=train_set_id, test_set_id=test_set_id)
    df_test_syn = df_test_orig.copy(deep=True)
    df_test_syn[input_column_name] = df_test_syn.apply(lambda x: replace_atoms_with_synonyms(x[input_column_name], atom_to_syn_dict), axis=1)
    df_train[input_column_name] = df_train.apply(lambda x: replace_atoms_with_synonyms(x[input_column_name], 
                                                            atom_to_syn_dict, p_replace=p_replace_train), axis=1)
    return df_train, df_test_orig, df_test_syn

#Localism
def get_df_test_localism(df_test, columns_to_exclude = ["Original_text"]):
    """Creates an alternative version of df_test by converting all input and output strings into lists of input and output items.

    Args:
        df_test (pd.DataFrame): DataFrame that contains both input and output sequences for the test set
        columns_to_exclude = List of columns to ignore when creating the localism set

    Returns:
        df_test_localism (pd.DataFrame): A duplicate of df_test where input and output strings have been converted into lists of items
    """
    df_test_localism = df_test.copy(deep=True)
    for col in df_test_localism.columns.tolist():
        if col not in columns_to_exclude:
            df_test_localism[col] = df_test_localism[col].str.split()
    return df_test_localism

#Overgeneralisation
# Does currently not work for compound words in Swedish, those get filtered out from both input and output
def replace_input_output_exception(input_str, output_str, exceptions_dict, language, from_unimorph_conversion_dict):
    """Applies exceptions specified in exceptions dict to the output string.

    Args:
        input_str (str): A single input sequence
        output_str (str): The corresponding output sequence
        exceptions_dict (dict): Dictionary with the original atom values as keys and dictionaries with keys "value" and "condition" 
                                as values where "value" contains the "exceptional" atom and "condition" shows which other atoms need
                                to be present for the exception to apply                      
        language (str): Language for creating the new output string. Currently valid values: Estonian, Swedish
        from_unimorph_conversion_dict (dict): Dictionary for converting UniMorph features to language-specific ones for synthesis

    Returns:
        
        new_input_str (str): New input sequence (currently only different from input_str if compound words occur in Swedish)
        old_output_str (str): Output sequence before applying the exception
        new_output_str (str): New output sequence with the exception applied
    """
    all_compounds = input_str.split()
    all_words = output_str.split()
    new_compounds = []
    new_words = []
    old_words = []
    for compound, word in zip(all_compounds, all_words):
        new_features_list = []
        features_list = compound.split("(")[0].split(";")
        lemma = compound.split("(")[1][:-1]
        new_features_list = [exceptions_dict[item]["value"] if (item in exceptions_dict.keys() and 
                             set(exceptions_dict[item]["condition"]).issubset(features_list)) else item
                             for item in features_list]
        new_features = ";".join(new_features_list)
        if features_list == new_features_list:
            new_compounds.append(compound)
            new_words.append(word)
            old_words.append(word)
        else:
            new_word_candidates = None
            if language == "Estonian":
                vabamorf_form = from_unimorph_conversion_dict[new_features]
                new_word_candidates = '|'.join(synthesize(lemma, vabamorf_form))
            elif language == "Swedish":
                convert_pos_dict = {"N" : "NN", "V" : "VB", "ADJ" : "JJ", "V.PTCP" : "JJ"}
                token_pos = convert_pos_dict[features_list[0]]
                token_complemgram = None #TODO: not applying exceptions to compound words atm
                token_compwf = None #TODO: not applying exceptions to compound words atm
                saldo_feats = convert_unimorph_feats_to_saldo(token_pos, new_features)
                new_word_candidates = get_possible_forms_saldo(lemma, token_pos, token_complemgram, token_compwf, 
                                                          saldo_feats, from_unimorph_conversion_dict)
            else:
                raise ValueError("Valid values for language: Estonian, Swedish")

            if new_word_candidates:
                new_compounds.append(f"{";".join(features_list)}({lemma})")
                new_words.append(new_word_candidates) 
                old_words.append(word)

    new_input_str = " ".join(new_compounds)
    old_output_str = " ".join(old_words)
    new_output_str = " ".join(new_words)
    return new_input_str, old_output_str, new_output_str

def get_exception_counts(df_data, input_column_name, exceptions_dict):
    """Counts the number of contexts the exceptions specified in exceptions_dict will be applied to.

    Args:
        df_data (pd.DataFrame): DataFrame containing all of the data
        input_column_name (str): Name of the column in df_data containing the input sequences
        exceptions_dict (dict): Dictionary with the original atom values as keys and dictionaries with keys "value" and "condition" 
                                as values where "value" contains the "exceptional" atom and "condition" shows which other atoms need
                                to be present for the exception to apply
    """
    
    for old, new in exceptions_dict.items():
        subset = [old] + new["condition"]
        count = 0
        for input_str in df_data[input_column_name].tolist():
            for item in input_str.split():
                if set(subset).issubset(item.split("(")[0].split(";")):
                    count += 1
        print(f"{"&".join(subset)} appears in {count} items.")

def apply_exception(df_data, input_column_name, output_column_name, exceptions_dict, language, conversion_dict, test_data=True):
    """Counts the number of contexts the exceptions specified in exceptions_dict will be applied to.

    Args:
        df_data (pd.DataFrame): DataFrame containing all of the data
        input_column_name (str): Name of the column in df_data containing the input sequences
        output_column_name (str): Name of the column in df_data containing the output sequences
        exceptions_dict (dict): Dictionary with the original atom values as keys and dictionaries with keys "value" and "condition" 
                                as values where "value" contains the "exceptional" atom and "condition" shows which other atoms need
                                to be present for the exception to apply
        language (str): Language for creating the new output string. Currently valid values: Estonian, Swedish
        conversion_dict (dict): Dictionary for converting UniMorph features to language-specific ones for synthesis

    Returns:
        df_data_temp (pd.DataFrame): Duplicate of df_data where the exceptions in exceptions_dict have been applied
    """
    df_data_temp = df_data.copy(deep=True)
    get_exception_counts(df_data, input_column_name, exceptions_dict)
    input_output = df_data_temp.apply(lambda x: replace_input_output_exception(x[input_column_name], x[output_column_name], exceptions_dict, language, conversion_dict), axis=1)
    old_output_column_name = f"{output_column_name}_original"
    df_data_temp[input_column_name], df_data_temp[old_output_column_name], df_data_temp[output_column_name] = zip(*input_output)
    return df_data_temp