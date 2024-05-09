import numpy as np
import torch
import glob
from tqdm.auto import tqdm
import os

full_data_list = torch.load('huge_list.pt')
training_list, validation_list, testing_list = torch.utils.data.random_split(full_data_list, [0.8, 0.1, 0.1])

def load_batches_raw(batch_size):
    data_groups = []
    print('Shuffling data (outside of batches!)')
    for data_group in [training_list, validation_list, testing_list]:
        random_indices = np.random.permutation(range(len(data_group)))
        batches = []
        for i in tqdm(range(0, len(data_group), batch_size)):
            end_index = min((i+batch_size), len(data_group))
            tensor_list = [torch.transpose(data_group[random_indices[j]],0,1) for j in range(i, end_index)]
            batch = torch.stack(tensor_list, dim=0)
            batches.append(batch)
        data_groups.append(batches)
    return tuple(data_groups)

def load_batches(batch_size):
    print('Batching...')
    return_tuple = []
    for data_group in ('training', 'validation', 'testing'):
        print(f'Loading data for {data_group} dataset')
        glob_path = os.path.join('data', data_group, '*.pt')
        all_data = []
        print('Reading in raw .pt data...')
        for fn in tqdm(glob.glob(glob_path)):
            all_data += (torch.load(fn)[::2])
        random_indices = np.random.permutation(range(len(all_data)))

        input_batches = []
        print('Batching...')
        for i in tqdm(range(0, len(all_data), batch_size)):
            end_index = min((i+batch_size), len(all_data))
            batched_inputs = [torch.transpose(all_data[random_indices[j]][0], 0, 1) for j in range(i, end_index)]
            input_batches.append(torch.stack(batched_inputs, dim=0))
        return_tuple.append(input_batches)
    print('\n')
    return tuple(return_tuple)

def load_batches_equity(batch_size):
    print('Batching...')
    return_tuple = []
    for data_group in ('training', 'validation', 'testing'):
        print(f'Loading data for {data_group} dataset')
        glob_path = os.path.join('data', data_group, '*.pt')
        all_data = []
        print('Reading in raw .pt data...')
        for fn in tqdm(glob.glob(glob_path)):
            all_data += (torch.load(fn))
        random_indices = np.random.permutation(range(len(all_data)))

        input_batches = []
        target_batches = []
        print('Batching...')
        for i in tqdm(range(0, len(all_data), batch_size)):
            end_index = min((i+batch_size), len(all_data))
            batched_inputs = [torch.transpose(all_data[random_indices[j]][0], 0, 1) for j in range(i, end_index)]
            batched_targets = [torch.tensor(all_data[random_indices[j]][1][0]) for j in range(i, end_index)]
            input_batches.append(torch.stack(batched_inputs, dim=0))
            target_batches.append(torch.stack(batched_targets, dim=0))
        return_tuple.append(input_batches)
        return_tuple.append(target_batches)

    print('\n')
    return tuple(return_tuple)