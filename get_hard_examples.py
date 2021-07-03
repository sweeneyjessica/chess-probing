import chess
import chess.svg
import sys
import torch
import random
import json
import os
from transformers import GPT2LMHeadModel, AutoModel
from error_analysis_end import analyze_error
from copy import deepcopy

sys.path.append("learning-chess-blindfolded/src")

from data_utils.chess_tokenizer import ChessTokenizer


def get_legal_moves(board):
    legal_moves = set()
    for move in board.legal_moves:
        uci_move = board.uci(move)
        legal_moves.add(uci_move)

    return legal_moves


def get_hard_ex(input_file):
    """Extract games from the evaluation test set in which the model predicted
     an illegal move, and write those game prefixes to a new file."""

    wrong_count = 0

    out_file = input_file.split(".")[0]
    out_file += "_wrong.jsonl"

    with open(out_file, "w") as output:
        with open(input_file) as reader:
            for line in reader:
                example = json.loads(line)
                board = chess.Board()

                game_prefix = [tokenizer.bos_token_id]

                for move in example["prefix"].split():
                    board.push_uci(move)
                    game_prefix.extend(tokenizer.encode(move, add_special_tokens=False, get_move_end_positions=False))

                start_square = example["starting_square"]
                legal_moves = example["legal_ending_square"]

                game_prefix.extend(tokenizer.encode(start_square, add_special_tokens=False, get_move_end_positions=False))

                pred_move = get_model_pred(game_prefix)[0]

                if pred_move not in legal_moves:
                    output.write(json.dumps(example))

                    arrows = [chess.svg.Arrow(chess.parse_square(start_square), chess.parse_square(pred_move))]
                    generate_image(board, arrows, wrong_count, "orig")

                    wrong_count += 1


def get_model_pred(game_prefix):
    # TODO: most other places assume get_model_pred is returning only top K
    """Returns sorted list of moves predicted by the model given a game prefix."""

    greedy_game_prefix = list(game_prefix)
    prefix_tens = torch.tensor([greedy_game_prefix])

    logits = model(prefix_tens)[0]
    last_token_logit = logits[0, -1, :]

    token_idx_sorted = torch.topk(last_token_logit, len(last_token_logit))[1]

    pred_moves = []

    for token in token_idx_sorted:
        token_idx = token.item()
        current_token = tokenizer.decode_token(token_idx)
        pred_moves.append(current_token)

    return pred_moves


def generate_image(board, arrows, idx, filepath):
    """Generates an SVG image of the board with the given arrows, and saves
    it using the filepath and index."""

    chessboard_svg = chess.svg.board(board, arrows=arrows)

    if not os.path.exists(f"{filepath}_images"):
        os.makedirs(f"{filepath}_images")

    img = open(f"{filepath}_images/{filepath}_board_{idx}.svg", "w")
    img.write(chessboard_svg)
    img.close()


def generate_topk_moves_images(top_k_preds, start_square, board, idx, filepath):

    arrows = []
    for arr, pred in enumerate(top_k_preds):
        if pred == tokenizer.eos_token:
            continue
        if arr == 0:
            arrows.append(chess.svg.Arrow(chess.parse_square(start_square), chess.parse_square(pred), color='green'))
        elif arr == 1:
            arrows.append(chess.svg.Arrow(chess.parse_square(start_square), chess.parse_square(pred), color='blue'))
        elif arr == 2:
            arrows.append(chess.svg.Arrow(chess.parse_square(start_square), chess.parse_square(pred), color='yellow'))
        else:
            arrows.append(chess.svg.Arrow(chess.parse_square(start_square), chess.parse_square(pred), color='red'))

    chessboardSvg = chess.svg.board(board, arrows=arrows)
    f = open("{}_images/{}_board_{}.svg".format(filepath, filepath, idx), "w")
    f.write(chessboardSvg)
    f.close()


def generate_empt_moves_images(top_k_preds, start_squares, board, idx, filepath):

    arrows = []
    for start, pred in zip(start_squares, top_k_preds):
        if pred == tokenizer.eos_token:
            continue

        arrows.append(chess.svg.Arrow(chess.parse_square(start), chess.parse_square(pred), color='green'))

    chessboardSvg = chess.svg.board(board, arrows=arrows)
    f = open("{}_images/{}_board_{}.svg".format(filepath, filepath, idx), "w")
    f.write(chessboardSvg)
    f.close()

def generate_arrow_on_board(start, end, board, idx, filepath):

    arrows = [chess.svg.Arrow(chess.parse_square(start), chess.parse_square(end), color='green')]

    chessboardSvg = chess.svg.board(board, arrows=arrows)
    f = open("{}_images/{}_board_{}.svg".format(filepath, filepath, idx), "w")
    f.write(chessboardSvg)
    f.close()

def get_path_obstructions(board, start, end):
    move = board.turn
    obstructions = []  # list of squares of obstructed pieces

    path_squares = chess.SquareSet.between(start, end)
    path_squares.add(end)

    for square in path_squares:
        if board.piece_at(square) is not None:
            if square is not end:
                obstructions.append(square)
            else:
                if board.color_at(square) == move:
                    obstructions.append(square)

    return obstructions


def empty_predictions(board, game_prefix_list, idx):
    empty_squares = []

    for square in chess.SquareSet(chess.BB_ALL):
        if board.piece_at(square) is None:
            empty_squares.append(square)

    random_empty = random.sample(empty_squares, 5)
    random_empty = [chess.square_name(x) for x in random_empty]

    empty_moves = []

    for square in random_empty:
        empt_prefix = deepcopy(game_prefix_list)

        empt_prefix.extend(tokenizer.encode(square, add_special_tokens=False, get_move_end_positions=False))

        empty_best_move = get_model_pred(empt_prefix, 1)[0]

        empty_moves.append(empty_best_move)

    generate_empt_moves_images(empty_moves, random_empty, board, idx, "empt")


def piece_predictions(board, game_prefix_list, idx):
    turn = board.turn

    all_side_pieces = chess.SquareSet(board.occupied_co[turn])
    all_side_pieces = [chess.square_name(x) for x in all_side_pieces]

    piece_moves = []

    for square in all_side_pieces:
        piece_prefix = deepcopy(game_prefix_list)

        piece_prefix.extend(tokenizer.encode(square, add_special_tokens=False, get_move_end_positions=False))

        piece_best_move = get_model_pred(piece_prefix, 1)[0]

        piece_moves.append(piece_best_move)

    generate_empt_moves_images(piece_moves, all_side_pieces, board, idx, "pieces")

    moves = []

    for st, en in zip(all_side_pieces, piece_moves):
        moves.append(st+en)

    return moves



def collect_info(input_file):
    with open(input_file) as reader:
        for idx, line in enumerate(reader):
            example = json.loads(line)

            game_prefix = example["prefix"].split()
            game_prefix_list = [tokenizer.bos_token_id]

            board = chess.Board()

            for move in game_prefix:
                board.push_uci(move)
                game_prefix_list.extend(tokenizer.encode(move, add_special_tokens=False, get_move_end_positions=False))


            # top K moves (maybe same # as # of legal moves)
            start_square = example["starting_square"]
            legal_moves = example["legal_ending_square"]

            original_query = deepcopy(game_prefix_list)
            original_query.extend(tokenizer.encode(start_square, add_special_tokens=False, get_move_end_positions=False))
            top_k_preds = get_model_pred(original_query, 5)

            #generate_topk_moves_images(top_k_preds, start_square, board, idx, "topk")

            top_pred = top_k_preds[0]
            best_move = start_square+top_pred

            if idx == 21:
                new_query = deepcopy(game_prefix_list)
                #new_query.extend(tokenizer.encode(start_square, add_special_tokens=False, get_move_end_positions=False))
                new_query.extend(tokenizer.encode("d8d5", add_special_tokens=False, get_move_end_positions=False))
                new_query.extend(tokenizer.encode("f4f5", add_special_tokens=False, get_move_end_positions=False))
                new_query.extend(tokenizer.encode("a4a3", add_special_tokens=False, get_move_end_positions=False))
                new_query.extend(tokenizer.encode("f3", add_special_tokens=False, get_move_end_positions=False))

                board.push(chess.Move.from_uci("d8d5"))
                board.push(chess.Move.from_uci("f4f5"))
                board.push(chess.Move.from_uci("a4a3"))

                pseud_top_k = get_model_pred(new_query, 5)
                generate_topk_moves_images(pseud_top_k, "f3", board, 21, "pseudo21")

            # bigram
            #bigram = game_prefix[-1:] + [best_move]
            #bigram_count = find_ngram_frequency(bigram)
            #print("Bigram freq. of last move + predicted move: {}".format(bigram_count))

            # trigram
            #trigram = game_prefix[-2:] + [best_move]
            #trigram_count = find_ngram_frequency(trigram)
            #print("Trigram freq. of last two moves + predicted move: {}".format(trigram_count))

            # ask about other pieces

            """piece_type = chess.piece_symbol(board.piece_type_at(chess.parse_square(start_square)))
            if analyze_error(board, piece_type, start_square, top_pred) == "Path Obstruction":
                start = chess.parse_square(start_square)
                end = chess.parse_square(top_pred)

                obstructions = get_path_obstructions(board, start, end)

                #circles = [(x, x) for x in obstructions]

                #chessboardSvg = chess.svg.board(board, arrows=circles)
                #f = open("obstruction_images/obs_board_{}.svg".format(idx), "w")
                #f.write(chessboardSvg)
                #f.close()

                obstructions = [chess.square_name(sq) for sq in obstructions]

                for sq in obstructions:
                    obs_prefix = deepcopy(game_prefix_list)

                    obs_prefix.extend(tokenizer.encode(sq, add_special_tokens=False, get_move_end_positions=False))

                    pseudo_legal_end_sq = []

                    for move in board.pseudo_legal_moves:
                        if move.from_square == chess.parse_square(sq):
                            pseudo_legal_end_sq.append(chess.square_name(move.to_square))

                    obs_top_k = get_model_pred(obs_prefix, 5)

                    valid_count = 0

                    for pred in obs_top_k:
                        if pred in pseudo_legal_end_sq:
                            valid_count += 1

                    generate_topk_moves_images(obs_top_k, sq, board, idx, "obs")

                    print("{} {}".format(idx, valid_count/5))"""



            # predictions from empty squares
            # empty_predictions(board, game_prefix_list, idx)

            # predictions for all pieces on board

            """all_side_moves = piece_predictions(board, game_prefix_list, idx)
            invalid = 0

            for side_move in all_side_moves:
                if chess.Move.from_uci(side_move) not in board.legal_moves:
                    invalid += 1

            print("Invalid moves: {} out of {}".format(invalid, len(all_side_moves)))"""

            """piece_type = chess.piece_symbol(board.piece_type_at(chess.parse_square(start_square)))
            if analyze_error(board, piece_type, start_square, top_pred) == "Pseudo Legal":
                original_query.extend(
                    tokenizer.encode(top_pred, add_special_tokens=False, get_move_end_positions=False))

                next_piece_st = get_model_pred(original_query, 1)[0]

                original_query.extend(
                    tokenizer.encode(next_piece_st, add_special_tokens=False, get_move_end_positions=False))

                next_piece_end = get_model_pred(original_query, 1)[0]

                uci_move = start_square+top_pred

                board.push(chess.Move.from_uci(uci_move))
                generate_arrow_on_board(next_piece_st, next_piece_end, board, idx, "pseudo")"""


def find_ngram_frequency():
    ngrams = []
    with open("end_long_wrong.jsonl") as examples:
        for line in examples:
            example = json.loads(line)

            game_prefix = example["prefix"].split()
            game_prefix_list = [tokenizer.bos_token_id]

            for move in game_prefix:
                game_prefix_list.extend(tokenizer.encode(move, add_special_tokens=False, get_move_end_positions=False))

            start_square = example["starting_square"]

            game_prefix_list.extend(
                tokenizer.encode(start_square, add_special_tokens=False, get_move_end_positions=False))

            top_pred = get_model_pred(game_prefix_list, 1)[0]

            best_move = start_square + top_pred

            # bigram
            bigram_denom = game_prefix[-1:] + [start_square]
            bigram = game_prefix[-1:] + [best_move]
            bigram = " ".join(bigram)
            bigram_denom = " ".join(bigram_denom)

            ngrams.append({"bigram": bigram, "denom":bigram_denom})

    with open("uci/train.txt") as train_data:
        counts = []
        for i in range(len(ngrams)):
            counts.append({"bigram": 0, "denom": 0})

        for line in train_data:
            for i, ex in enumerate(ngrams):
                counts[i]["bigram"] += line.count(ex["bigram"])
                counts[i]["denom"] += line.count(ex["denom"])

    with open("ngram_freq_percent.txt", "w") as f:
        for idx, example in enumerate(counts):
            if example["bigram"] == 0 or example["denom"] == 0:
                f.write("{} 0\n".format(idx))
            else:
                f.write("{} {}\n".format(idx, (example["bigram"]/example["denom"])*100))


def testing_ex_six():
    with open("example_six.jsonl") as example:
        six = json.loads(example.readline())

    game_prefix = six["prefix"].split()
    game_prefix_list = [tokenizer.bos_token_id]

    board = chess.Board()

    for move in game_prefix:
        board.push_uci(move)
        game_prefix_list.extend(tokenizer.encode(move, add_special_tokens=False, get_move_end_positions=False))

    pred_from = get_model_pred(game_prefix_list, 1)[0]

    game_prefix_list.extend(tokenizer.encode(pred_from, add_special_tokens=False, get_move_end_positions=False))

    pred_to = get_model_pred(game_prefix_list, 1)[0]

    board.push_uci(pred_from+pred_to)

    chessboardSvg = chess.svg.board(board, arrows=[chess.svg.Arrow(chess.parse_square(pred_from), chess.parse_square(pred_to))])
    f = open("six_pred_whole_move.svg", "w")
    f.write(chessboardSvg)
    f.close()


def show_board_given_uci(uci, stop_ngram, num):
    board = chess.Board()

    stop = 0
    for move in uci.split():
        if move == stop_ngram[0]:
            stop += 1
            board.push_uci(move)
            continue
        if move == stop_ngram[1] and stop:
            break

        board.push_uci(move)

    move1 = chess.Move.from_uci(stop_ngram[0])
    move2 = chess.Move.from_uci(stop_ngram[1])

    chessboardSvg = chess.svg.board(board, arrows=[
        chess.svg.Arrow(move1.from_square, move1.to_square, color='red'), chess.svg.Arrow(move2.from_square, move2.to_square, color='green'),])
    f = open("{}_{}_{}.svg".format(stop_ngram[0], stop_ngram[1], num), "w")
    f.write(chessboardSvg)
    f.close()



if __name__ == "__main__":
    """### Initialize Model and Tokenizer"""

    vocab_path = "learning-chess-blindfolded/sample_data/lm_chess/vocab/uci/vocab.txt"
    tokenizer = ChessTokenizer(vocab_path)

    model = GPT2LMHeadModel.from_pretrained('shtoshni/gpt2-chess-uci')

    #get_hard_ex("end_long.jsonl")

    collect_info("end_long_wrong.jsonl")

    #testing_ex_six()

    #find_ngram_frequency()

    #show_board_given_uci("d2d4 g8f6 c2c4 e7e6 b1c3 f8b4 d1c2 d7d5 a2a3 b4c3 c2c3 c7c5 d4c5 d5d4 c3c2 a7a5 g1f3 b8c6 c1f4 e8g8 a1d1 f8e8 f3e5 f6d7 e5c6 b7c6 f4d6 e6e5 b2b4 a5b4 a3b4 e5e4 d1d4 a8a1 d4d1 d8f6 e2e3 a1d1 c2d1 f6b2 f1e2 b2b4 d1d2 b4b1 d2d1 b1b4 d1d2 b4b1 e2d1 c8a6 e1g1 a6c4 f1e1 e8a8 d1g4 a8a1", ["a8a1","e1a1"], "orig")

