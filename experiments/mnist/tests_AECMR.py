import unittest
from sklearn.metrics import accuracy_score
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

from experiments.mnist.mnist_dataset import addition_dataset
from experiments.mnist.autoencoderCMR import (
    MNISTModel, MNISTEncoder, AECat,
    get_accuracy
)

checkpoint_path = './results/mnist_base/CMR/best.ckpt'


NUM_DIGITS   = 2
DIGIT_LIMIT  = 10
EMB_SIZE     = 500
RULE_EMB     = 1000
N_RULES      = 20
LR           = 0.0001
BATCH_SIZE   = 512
MAX_EPOCHS   = 200
VAL_SPLIT    = 0.1
VERBOSE = False


def c_idx_to_name(c_idx, num_digits, digit_limit):
    d = c_idx // digit_limit
    v = c_idx % digit_limit
    return f"d{d}_{v}"


def get_mnist_addition_loaders(batch_size, val_split=0.1, shuffle_train=True):
    x_train_raw, c_train_raw, y_train_raw = addition_dataset(True, NUM_DIGITS, DIGIT_LIMIT)
    x_test_raw,  c_test_raw,  y_test_raw  = addition_dataset(False, NUM_DIGITS, DIGIT_LIMIT)

    x_train = torch.stack(x_train_raw, dim=1).float()
    c_train = torch.cat(c_train_raw, dim=-1).float()
    y_train = F.one_hot(y_train_raw.unsqueeze(-1).long().ravel()).float()

    x_test = torch.stack(x_test_raw, dim=1).float()
    c_test = torch.cat(c_test_raw, dim=-1).float()
    y_test = F.one_hot(y_test_raw.unsqueeze(-1).long().ravel()).float()

    split = int(len(x_train) * (1 - val_split))
    x_val,  c_val,  y_val  = x_train[split:], c_train[split:], y_train[split:]
    x_train, c_train, y_train = x_train[:split], c_train[:split], y_train[:split]

    train_loader = DataLoader(TensorDataset(x_train, c_train, y_train), batch_size=batch_size, shuffle=shuffle_train)
    val_loader   = DataLoader(TensorDataset(x_val,   c_val,   y_val),   batch_size=batch_size)
    test_loader  = DataLoader(TensorDataset(x_test,  c_test,  y_test),  batch_size=batch_size)

    return train_loader, val_loader, test_loader, c_train.shape[1], y_train.shape[1]


def get_model():
    _, _, test_loader, n_concepts, n_tasks = get_mnist_addition_loaders(
            BATCH_SIZE, VAL_SPLIT
        )

        
    model = MNISTModel.load_from_checkpoint(
        checkpoint_path,
        encoder=MNISTEncoder(
            emb_size=EMB_SIZE,
            cp_output=DIGIT_LIMIT,
            number_digits=NUM_DIGITS
        ),
        rule_module=AECat,
        weights_only=False
    )

    model.eval()
    model.make_editable()

    return model, test_loader, n_concepts, n_tasks



class AECMRTest(unittest.TestCase):

    def test_add_good_rules(self):
        print("=== ADD GOOD RULES ===")

        model, test_loader, n_concepts, n_tasks = get_model()

        good_pairs_per_task = {
            0:  [(0,0)],
            1:  [(0,1), (1,0)],
            2:  [(0,2), (1,1), (2,0)],
            3:  [(0,3), (1,2), (2,1), (3,0)],
            4:  [(0,4), (1,3), (2,2), (3,1), (4,0)],
            5:  [(0,5), (1,4), (2,3), (3,2), (4,1), (5,0)],
            6:  [(0,6), (1,5), (2,4), (3,3), (4,2), (5,1), (6,0)],
            7:  [(0,7), (1,6), (2,5), (3,4), (4,3), (5,2), (6,1), (7,0)],
            8:  [(0,8), (1,7), (2,6), (3,5), (4,4), (5,3), (6,2), (7,1), (8,0)],
            9:  [(0,9), (1,8), (2,7), (3,6), (4,5), (5,4), (6,3), (7,2), (8,1), (9,0)],
            10: [(1,9), (2,8), (3,7), (4,6), (5,5), (6,4), (7,3), (8,2), (9,1)],
            11: [(2,9), (3,8), (4,7), (5,6), (6,5), (7,4), (8,3), (9,2)],
            12: [(3,9), (4,8), (5,7), (6,6), (7,5), (8,4), (9,3)],
            13: [(4,9), (5,8), (6,7), (7,6), (8,5), (9,4)],
            14: [(5,9), (6,8), (7,7), (8,6), (9,5)],
            15: [(6,9), (7,8), (8,7), (9,6)],
            16: [(7,9), (8,8), (9,7)],
            17: [(8,9), (9,8)],
            18: [(9,9)],
        }

        @torch.no_grad()
        def get_task_accuracy(task_id):
            preds_list, y_list = [], []

            for x, c, y in test_loader:
                y_pred = model.predict((x, c, y))

                preds_list.append(y_pred[:, task_id].cpu())
                y_list.append(y[:, task_id].cpu())

            preds = torch.cat(preds_list).numpy()
            ys = torch.cat(y_list).numpy()

            return accuracy_score(ys, preds)
        
        baseline_global = get_accuracy(model, test_loader) 

        results = []

        with torch.no_grad():
            for t in range(n_tasks):

                print(f"================ TASK {t} ================")

                baseline_task = get_task_accuracy(t)
                good_rules = good_pairs_per_task[t]

                for g in good_rules:

                    d1, d2 = g

                    rule = torch.zeros((n_concepts, 3), device=model.rule_module.rules.weight.device)
                    rule[:, 1] = 1.0
                    rule[d1] = torch.tensor([1.,0.,0.], device=rule.device)
                    rule[d2 + 10] = torch.tensor([1.,0.,0.], device=rule.device)

                    success = model.add_rule(t, rule)

                    if success:
                        print(f"Added rule {g}")

                    else:
                        print(f"Failed to add rule {g}")
                        continue

                mod_global = get_accuracy(model, test_loader) 
                mod_task = get_task_accuracy(t)

                results.append({
                    "task": t,
                    "baseline_global": baseline_global,
                    "modified_global": mod_global,
                    "global_delta": mod_global - baseline_global,
                    "baseline_task": baseline_task,
                    "modified_task": mod_task,
                    "task_delta": mod_task - baseline_task,
                })

                model = MNISTModel.load_from_checkpoint(
                    checkpoint_path,
                    encoder=MNISTEncoder(
                        emb_size=EMB_SIZE,
                        cp_output=DIGIT_LIMIT,
                        number_digits=NUM_DIGITS
                    ),
                    rule_module=AECat,
                    weights_only=False
                )

                model.eval()
                model.make_editable()

        pd.DataFrame(results).to_csv(f"results/rule_addition/add_good_rules.csv", index=False)


    def test_delete_bad_rules(self):

        print("=== DELETE BAD RULES  ===")

        model, test_loader, n_concepts, n_tasks = get_model()

        bad_rules = {
            0:  [0,2,3,4,5,7,8,9,10,12,13,15,16,17,18,19],
            1:  [0,1,2,3,4,6,7,8,10,11,13,15,17],
            2:  [0,1,3,4,5,7,8,10,12,13,18,19],
            3:  [6,9,19],
            4:  [6,8,10,11,17],
            5:  [0,2,9,13,15,16],
            6:  [1,3,11,16,17,18,19],
            7:  [0,2,3,7,11,16,17],
            8:  [1,2,3,10,11],
            9:  [4,10,14],
            10: [5,10],
            11: [10,11,13],
            12: [4,6,7,9],
            13: [0,6,12,14,18],
            14: [8,9,11,12,14,16],
            15: [1,5,12,13,15,18,19],
            16: [1,3,5,7,10],
            17: [3,4,6,8,9,12,13,14,16,17],
            18: [2,4,6,7,9,10,11,12,13,14,15,16,17,18,19],
        }

        @torch.no_grad()
        def get_task_accuracy(task_id):
            preds_list, y_list = [], []

            for x, c, y in test_loader:
                y_pred = model.predict((x, c, y))

                preds_list.append(y_pred[:, task_id].cpu())
                y_list.append(y[:, task_id].cpu())

            preds = torch.cat(preds_list).numpy()
            ys = torch.cat(y_list).numpy()

            return accuracy_score(ys, preds)

        baseline_global = get_accuracy(model, test_loader) 

        results = []


        with torch.no_grad():

            for t in range(n_tasks):

                print(f"================ TASK {t} ================")

                baseline_task = get_task_accuracy(t)
                to_delete = sorted(bad_rules[t], reverse=True)

                print(f"Deleting rules: {to_delete}")

                for rid in to_delete:

                    success = model.delete_rule(t, rid)
                        
                    if success:
                        print(f"Deleted rule {rid}")

                    else:
                        print(f"Failed to delete rule {rid}")


                mod_global = get_accuracy(model, test_loader) 
                mod_task = get_task_accuracy(t)

                results.append({
                    "task": t,
                    "baseline_global": baseline_global,
                    "modified_global": mod_global,
                    "global_delta": mod_global - baseline_global,
                    "baseline_task": baseline_task,
                    "modified_task": mod_task,
                    "task_delta": mod_task - baseline_task,
                    "num_deleted": len(to_delete),
                })

                print(
                    f"TASK {t}: "
                    f"{baseline_task:.4f} -> {mod_task:.4f}"
                )


                model = MNISTModel.load_from_checkpoint(
                    checkpoint_path,
                    encoder=MNISTEncoder(
                        emb_size=EMB_SIZE,
                        cp_output=DIGIT_LIMIT,
                        number_digits=NUM_DIGITS
                    ),
                    rule_module=AECat,
                    weights_only=False
                )

                model.eval()
                model.make_editable()

        pd.DataFrame(results).to_csv(f"results/rule_deletion/delete_bad_rules.csv", index=False)


if __name__ == '__main__':

    AECMRTest().test_add_good_rules()
    AECMRTest().test_delete_bad_rules()