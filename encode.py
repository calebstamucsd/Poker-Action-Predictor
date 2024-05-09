import numpy as np
import os
import torch
import json
from sklearn.preprocessing import OneHotEncoder
from tqdm.auto import tqdm
import glob

READ_FILE = 'hands_valid.json'

RANKMAP = {'A':1, 'T':10, 'J':11, 'Q':12, 'K':13}
SUITMAP = {'c':1, 'd':2, 'h':3, 's':4}

def partial_encode_hand(card_list, one_hot_suits=False, one_hot_ranks=False):
    """
    Converts a list of cards in string format to an encoding
    
    parameters:
    card_list: list of strings
    one_hot_suits: false to encode cards densely but numerically, true to one-hot encode (sparse)
    one_hot_ranks: same as suits

    returns:
    a matrix of integers where each row represents one card, encoded as specified via parameters
    """
    rank_list = [card[0] for card in card_list]
    suit_list = [card[1] for card in card_list]
    
    def encode_rank_OHE(ranks):
        encoder = OneHotEncoder(categories=([['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T','J','Q','K']]))
        return encoder.fit_transform(np.array(ranks).reshape(-1,1)).toarray()
    def encode_suit_OHE(suits):
        encoder = OneHotEncoder(categories=([['c','d','h','s']]))
        return encoder.fit_transform(np.array(suits).reshape(-1,1)).toarray()
    def encode_rank_numerical(ranks):
        return np.vectorize(lambda x: x if x not in RANKMAP.keys() else RANKMAP[x])(np.array(ranks)).reshape(-1,1)
    def encode_suit_numerical(suits):
        return np.vectorize(lambda x: SUITMAP[x])(np.array(suits)).reshape(-1,1)
    
    if(one_hot_ranks):
        encoded_ranks = encode_rank_OHE(rank_list).astype(int)
    else:
        encoded_ranks = encode_rank_numerical(rank_list).astype(int)

    if(one_hot_suits):
        encoded_suits = encode_suit_OHE(suit_list).astype(int)
    else:
        encoded_suits = encode_suit_numerical(suit_list).astype(int)

    return np.concatenate((encoded_ranks, encoded_suits), axis=1)

def encode_board(card_list, rounds, full_ohe=True):
    '''
    Returns an encoding of the board that hides cards that the dealer hasn't revealed yet

    Parameters:
    card_list: list of all five cards on the board
    rounds: one-hot encoding of the game's rounds. rows equal to number of rounds, 4 columns (for pre-flop, flop, turn, river)
    one_hot_ranks, one_hot_suits: see encode_hand

    Returns:
    submatrix ready to be added to the overall encoding. Number of rows depends on embedding choice, number of columns
    equal to the number of rounds
    '''
    full_board = encode_hand(card_list, full_ohe=full_ohe)
    board_list = []
    for round in rounds:
        if(round[0] == 1):
            board_list.append(np.zeros(full_board.shape).flatten().reshape(-1, 1))
        elif(round[1] == 1):
            flop = full_board[:3]
            hidden = np.zeros((2, full_board.shape[1]))
            board_list.append(np.concatenate((flop, hidden)).flatten().reshape(-1,1))
        elif(round[2] == 1):
            flop_turn = full_board[:4]
            hidden = np.zeros((1, full_board.shape[1]))
            board_list.append(np.concatenate((flop_turn, hidden)).flatten().reshape(-1,1))
        else:
            board_list.append(full_board.flatten().reshape(-1,1))
    return np.concatenate(board_list, axis=1)

def encode_hand(card_list, full_ohe=True):
    if(full_ohe):
        # Define all possible cards in a sorted order
        suits = 'cdhs'  # clubs, diamonds, hearts, spades
        ranks = 'A23456789TJQK'
        all_cards = [rank + suit for rank in ranks for suit in suits]
        card_encoder = OneHotEncoder(categories=[all_cards])
        return card_encoder.fit_transform(np.array(card_list).reshape(-1,1)).toarray()
    else:
        return partial_encode_hand(card_list)


def encode_bets(bets):
    '''
    One-hot encodes the rounds of the game (pre-flop, flop, turn, and river) as well as the many
    actions a SINGLE player has made. i.e., we need to call this once per player
    '''
    round_encoder = OneHotEncoder(categories=[['p', 'f', 't', 'r']])
    action_encoder = OneHotEncoder(categories=[['-', 'B', 'f', 'k', 'b', 'c', 'r', 'A', 'Q', 'K']])
    round_list, action_list = [], []
    for round in bets:
        for action in round['actions']:
            round_list.append(round['stage'])
            action_list.append(action)
    rounds = round_encoder.fit_transform(np.array(round_list).reshape(-1, 1)).toarray()
    actions = action_encoder.fit_transform(np.array(action_list).reshape(-1,1)).toarray()
    return rounds, actions

def extract_data(read_file='hands_valid.json', full_ohe=True, two_only=True, num_per_file=5000):
    '''
    Takes in json data about a game of poker and returns an encoded version of the full game
    One columns per "cycle around the table". A single column contains many embeddings for different game properties.
    '''
    data_list = []
    target_list = []
    with open('hands_valid.json', 'r') as file:
        counter = 1
        for line in tqdm(file): # Each line is one game
            data = json.loads(line)
            if(two_only and (data['num_players'] == 2)):
                p1_bets, p2_bets = data['players'][0]['bets'], data['players'][0]['bets'] #All the hard information
                rounds, p1_actions = encode_bets(p1_bets)
                _, p2_actions = encode_bets(p2_bets)
                num_rounds = rounds.shape[0]

                encoded_board = (encode_board(data['board'], rounds, full_ohe=full_ohe)).T

                money_features = np.array([data['players'][player][feature] for feature in ['bankroll', 'action', 'winnings'] 
                                        for player in [0,1]]).reshape(6, 1).T

                encoded_p1_pocket_num = partial_encode_hand(data['players'][0]['pocket_cards']).flatten().reshape(1,-1)
                repeated_p1_pocket_num = np.repeat(encoded_p1_pocket_num, num_rounds, axis=0)
                encoded_p2_pocket_num = partial_encode_hand(data['players'][1]['pocket_cards']).flatten().reshape(1,-1)
                repeated_p2_pocket_num = np.repeat(encoded_p2_pocket_num, num_rounds, axis=0)

                encoded_p1_pocket_oh = encode_hand(data['players'][0]['pocket_cards']).flatten().reshape(1,-1)
                encoded_p2_pocket_oh = encode_hand(data['players'][1]['pocket_cards']).flatten().reshape(1,-1)

                money_features = np.repeat(money_features, num_rounds, axis=0)

                pots = np.array([x['size'] for x in data['pots']]) # encodes pot size per round (so always size 4)
                pots = (rounds @ pots).reshape(-1,1) # extends this to be one per "turn" instead of once per round

                data_list.append(torch.from_numpy(np.concatenate((encoded_board, pots, rounds, p1_actions, p2_actions, money_features,
                                            repeated_p1_pocket_num), axis=1)))
                data_list.append(torch.from_numpy(np.concatenate((encoded_board, pots, rounds, p1_actions, p2_actions, money_features,
                                            repeated_p2_pocket_num), axis=1)))
                target_list.append(torch.from_numpy(encoded_p2_pocket_oh))
                target_list.append(torch.from_numpy(encoded_p1_pocket_oh))

            if(counter % 6 == 5):
                dataset = 'validation'
            elif(counter % 6 == 4):
                dataset = 'test'
            else:
                dataset='training'

            # if(counter <= 100):
            #     dataset = 'training'
            # elif(counter <= 115):
            #     dataset = 'validation'
            # else:
            #     dataset = 'test'

            if(full_ohe):
                out_path = os.path.join('data', 'fully_encoded', dataset)
            else:
                out_path = os.path.join('data', 'partial_encoded', dataset)
            if not os.path.exists(out_path):
                os.makedirs(out_path)

            if(len(data_list) >= num_per_file):
                print(f'Writing data to files: Batch {counter}')
                torch.save(data_list, os.path.join(out_path, f'input_data_{counter:03}.pt'))
                torch.save(target_list, os.path.join(out_path, f'target_data_{counter:03}.pt'))
                data_list = []
                target_list = []
                counter += 1
        print(f'Writing data to files: Batch {counter}')
        torch.save(data_list, os.path.join(out_path, f'input_data_{counter:03}.pt'))
        torch.save(target_list, os.path.join(out_path, f'target_data_{counter:03}.pt'))




# Run the function
extract_data(full_ohe=False)