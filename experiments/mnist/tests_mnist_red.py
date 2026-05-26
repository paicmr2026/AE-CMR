from pathlib import Path
from itertools import product
from collections import defaultdict

import pandas as pd

import torch
import torchvision
from sklearn.metrics import accuracy_score

from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import TensorDataset, DataLoader

import torch.nn.functional as F
import lightning.pytorch as pl


from autoencoderCMR_red import (
    MNISTModel,
    MNISTEncoder,
    AECat,
    get_accuracy,
)

EMB_SIZE     = 500
RULE_EMB     = 1000
N_RULES      = 21

LR           = 0.0001
BATCH_SIZE   = 512
MAX_EPOCHS   = 200
VAL_SPLIT    = 0.1
SEED         = 10

TRAIN_PROB = 0.3
TEST_PROB = 0.3

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

data_root = Path("./data")

checkpoint_path = './experiments/mnist/results/mnist_red/best.ckpt'

def c_idx_to_name(c_idx, num_digits, digit_limit):

    block_size = digit_limit + 1

    d = c_idx // block_size
    v = c_idx % block_size

    if v == digit_limit:
        return f"d{d}red"

    return f"d{d}_{v}"


def rule_to_symbolic_filtered(rule, num_digits, digit_limit):

    block_size = digit_limit + 1

    parts = []

    for c_idx in range(rule.shape[0]):

        d = c_idx // block_size
        v = c_idx % block_size

        if v == digit_limit:
            name = f"d{d}red"
        else:
            name = f"d{d}_{v}"

        if rule[c_idx, 0] > 0.5:
            parts.append(f"{name}+")
        elif rule[c_idx, 2] > 0.5:
            parts.append(f"{name} irr")

    return " & ".join(parts) if parts else "∅"


def get_mnist_data(train: bool) -> MNIST:
    return torchvision.datasets.MNIST(
        root=str(data_root),
        train=train,
        download=True,
        transform=transform
    )


def colorize_tensor(images, red_mask):
    """
    Convert grayscale MNIST tensors into RGB tensors.

    Args:
        images:
            Tensor (N,1,H,W)

        red_mask:
            Bool tensor (N,)
            True -> RED
            False -> GRAY

    Returns:
        Tensor (N,3,H,W)
    """

    N, _, H, W = images.shape

    rgb = torch.zeros((N, 3, H, W), dtype=images.dtype)

    # Default GRAY
    rgb[:, 0] = images[:, 0]
    rgb[:, 1] = images[:, 0]
    rgb[:, 2] = images[:, 0]

    # RED samples
    rgb[red_mask, 1] = 0
    rgb[red_mask, 2] = 0

    return rgb


def apply_color_bias(
    X,
    y,
    train=True,
    train_red_prob=1.0,
    test_red_prob=0.5
):
    """
    Apply spurious color bias.

    TRAIN:
        sum == 12 --> RED with probability train_red_prob

    TEST:
        random RED with probability test_red_prob
    """

    N = len(y)

    if train:

        # Candidates
        sum12_mask = (y == 12)

        # Random probability mask
        random_mask = (
            torch.rand(N) < train_red_prob
        )

        # Only some SUM=12 become RED
        red_mask = sum12_mask & random_mask

    else:

        # Random RED during testing
        red_mask = (
            torch.rand(N) < test_red_prob
        )

    X_colored = []

    for digit_tensor in X:

        rgb_tensor = colorize_tensor(
            digit_tensor,
            red_mask
        )

        X_colored.append(rgb_tensor)

    return tuple(X_colored), red_mask


def addition_dataset_red(
    train,
    num_digits,
    digit_limit=10,
    train_red_prob=1.0,
    test_red_prob=0.5
):
    """
    Creates MNIST addition dataset.

    Returns:
        X           tuple of RGB digit tensors
        c           concept tensors
        y           sum labels
        red_mask    RED concept
    """

    dataset = get_mnist_data(train)

    X, y = dataset.data, dataset.targets

    # Restrict digits
    X = X[y < digit_limit]
    y = y[y < digit_limit]

    # Shape: (N,1,28,28)
    X = torch.unsqueeze(X, 1).float() / 255.0

    size = len(X) // num_digits

    X = torch.split(X, size)
    y = torch.split(y, size)

    if len(X) % num_digits != 0:
        X = X[:-1]
        y = y[:-1]

    # +1 for RED concept
    c = [
        torch.zeros((len(X[0]), digit_limit + 1)).float()
        for _ in range(len(X))
    ]

    for i, ys in enumerate(y):
        for j, yi in enumerate(ys):

            # One-hot digit concept
            c[i][j, yi] = 1.0

    y = torch.sum(torch.stack(y, 0), 0)

    X, red_mask = apply_color_bias(
        X,
        y,
        train=train,
        train_red_prob=train_red_prob,
        test_red_prob=test_red_prob
    )

    for i in range(len(c)):

        # Last column = RED concept
        c[i][:, -1] = red_mask.float()

    return X, c, y, red_mask

def create_single_digit_addition(
    num_digits,
    digit_limit=10
):

    concept_names = [
        "x%d%d" % (i, j)
        for i, j in product(
            range(num_digits),
            range(digit_limit)
        )
    ]

    # Add RED concept per digit
    concept_names.extend([
        f"x{i}RED"
        for i in range(num_digits)
    ])

    sums = defaultdict(list)

    for d in product(
        *[range(digit_limit)
          for _ in range(num_digits)]
    ):

        conj = []
        z = 0

        for i, n in enumerate(d):

            conj.append(
                "x%d%d" % (i, n)
            )

            z += n

        sums[z].append(
            "(" + " & ".join(conj) + ")"
        )

    explanations = {}

    class_names = [
        "z%d" % z
        for z in range(
            digit_limit * num_digits
            - num_digits + 1
        )
    ]

    for z in range(
        digit_limit * num_digits
        - num_digits + 1
    ):

        explanations["z%d" % z] = {
            "name": "%d" % z,
            "explanation":
                "(" + " | ".join(sums[z]) + ")"
        }

    return (
        concept_names,
        class_names,
        explanations
    )




class AECMR_Red:

    def test_red_irr(self):

        number_digits = 2

        concept_names, class_names, explanations = \
            create_single_digit_addition(number_digits)
        
        print(concept_names)

        
        _, _, y_train, _ = addition_dataset_red(
        train=True,
        num_digits=number_digits,
        train_red_prob=TRAIN_PROB,
        test_red_prob=TEST_PROB
        )

        X_test, c_test, y_test, _ = addition_dataset_red(
        train=False,
        num_digits=number_digits,
        train_red_prob=TRAIN_PROB,
        test_red_prob=TEST_PROB
        )


        pl.seed_everything(SEED)

        x_test_tensor  = torch.stack(X_test, dim=1).float()
        c_test_tensor  = torch.cat(c_test, dim=-1).float()
        y_train_tensor = F.one_hot(y_train.long()).float()
        y_test_tensor  = F.one_hot(y_test.long()).float()


        test_loader = DataLoader(
            TensorDataset(
                x_test_tensor,
                c_test_tensor,
                y_test_tensor
            ),
            batch_size=BATCH_SIZE
        )

        n_tasks = y_train_tensor.shape[1]

        model = MNISTModel.load_from_checkpoint(
            checkpoint_path,
            encoder=MNISTEncoder(
                emb_size=EMB_SIZE,
                cp_output=11,
                number_digits=number_digits
            ),
            rule_module=AECat,
            weights_only=False
        )

        model = model.cpu()

        model.eval()
        model.make_editable()

        @torch.no_grad()
        def eval_global():
            return get_accuracy(model, test_loader)

        @torch.no_grad()
        def eval_task(task_id):
            model.eval()

            preds_list = []
            y_list = []

            with torch.no_grad():
                for x, c, y in test_loader:
                    y_pred = model.predict((x, c, y))

                    preds_task = y_pred[:, task_id]
                    y_task = y[:, task_id]

                    preds_list.append(preds_task.cpu())
                    y_list.append(y_task.cpu())

            preds = torch.cat(preds_list, dim=0).numpy()
            ys = torch.cat(y_list, dim=0).numpy()

            return accuracy_score(ys, preds)
        
        baseline_global = eval_global()

        results = []

        with torch.no_grad():

            for t in range(n_tasks):

                print(f"================ TASK {t} ================")

                baseline_task = eval_task(t)

                rules = model.get_all_rule_vars()[t]
                print(rules.shape)

                for i in range(len(rules)):

                    rule = rules[i].clone()

                    rule[10] = torch.tensor(
                                    [0., 0., 1.],
                                    device=rule.device
                                )

                    rule[21] = torch.tensor(
                                    [0., 0., 1.],
                                    device=rule.device
                                )
                    
                    success = model.change_rule(t, i, rule)

                    if success:
                        print(f"Set red to irrelevant for rule {i} of task {t}")
                    else:
                        print(f"Failed to set red to irrelevant for rule {i} of task 12")

                modified_global = eval_global()
                modified_task = eval_task(t)

                print("Baseline global: ", baseline_global)
                print("Modified global :", modified_global)

                print("Baseline task accuracy :", baseline_task)
                print("Modified task accuracy :", modified_task)
                
                results.append({
                    "task_id": t,
                    "baseline_global": baseline_global,
                    "modified_global": modified_global,
                    "baseline_task": baseline_task,
                    "modified_task": modified_task,
                    "global_delta": modified_global - baseline_global,
                    "task_delta": modified_task - baseline_task,
                })

                model = MNISTModel.load_from_checkpoint(
                    checkpoint_path,
                    encoder=MNISTEncoder(
                        emb_size=EMB_SIZE,
                        cp_output=11,
                        number_digits=number_digits
                    ),
                    rule_module=AECat,
                    weights_only=False
                )

                model = model.cpu()

                model.eval()
                model.make_editable()

            pd.DataFrame(results).to_csv("results/mnist_red_bias/red_irr.csv", index=False)

        
    def test_corr_rules_red_irr(self):

        good_rules = {
            0: [16, 18, 19],
            1: [1, 4, 13, 17],
            2: [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 18, 20],
            3: [6, 8, 9, 10, 13, 14, 15, 16, 19, 20],
            4: [1, 2, 3, 4, 5, 7, 8, 9, 15, 17],
            5: [1, 2, 4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 19, 20],
            6: [0, 1, 3, 5, 6, 8, 9, 10, 11, 12, 14, 16, 17, 19, 20],
            7: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            8: [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20],
            9: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 16, 17, 18, 19],
            10: [0, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19],
            11: [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            12: [0, 1, 2, 3, 5, 6, 7, 9, 10, 12, 13, 14, 18, 19, 20],
            13: [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 20],
            14: [0, 1, 2, 3, 4, 5, 8, 9, 12, 13, 14, 16, 17, 18, 19, 20],
            15: [0, 1, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            16: [4, 8, 9, 11, 13, 14, 15, 16, 19],
            17: [0, 1, 3, 4, 6, 7, 8, 9, 10, 14, 15, 16, 20],
            18: [3, 4, 5, 9, 10, 14]
        }

        number_digits = 2

        concept_names, _, _ = create_single_digit_addition(number_digits)
        
        print(concept_names)

        _, _, y_train, _ = addition_dataset_red(
        train=True,
        num_digits=number_digits,
        train_red_prob=TRAIN_PROB,
        test_red_prob=TEST_PROB
        )

        X_test, c_test, y_test, _ = addition_dataset_red(
        train=False,
        num_digits=number_digits,
        train_red_prob=TRAIN_PROB,
        test_red_prob=TEST_PROB
        )

        pl.seed_everything(SEED)

        x_test_tensor  = torch.stack(X_test, dim=1).float()
        c_test_tensor  = torch.cat(c_test, dim=-1).float()
        y_train_tensor = F.one_hot(y_train.long()).float()
        y_test_tensor  = F.one_hot(y_test.long()).float()


        test_loader = DataLoader(
            TensorDataset(
                x_test_tensor,
                c_test_tensor,
                y_test_tensor
            ),
            batch_size=BATCH_SIZE
        )

        n_tasks = y_train_tensor.shape[1]

        model = MNISTModel.load_from_checkpoint(
            checkpoint_path,
            encoder=MNISTEncoder(
                emb_size=EMB_SIZE,
                cp_output=11,
                number_digits=number_digits
            ),
            rule_module=AECat,
            weights_only=False
        )

        model = model.cpu()

        model.eval()
        model.make_editable()

        @torch.no_grad()
        def eval_global():
            return get_accuracy(model, test_loader)

        @torch.no_grad()
        def eval_task(task_id):
            model.eval()

            preds_list = []
            y_list = []

            with torch.no_grad():
                for x, c, y in test_loader:
                    y_pred = model.predict((x, c, y))

                    preds_task = y_pred[:, task_id]
                    y_task = y[:, task_id]

                    preds_list.append(preds_task.cpu())
                    y_list.append(y_task.cpu())

            preds = torch.cat(preds_list, dim=0).numpy()
            ys = torch.cat(y_list, dim=0).numpy()

            return accuracy_score(ys, preds)
        
        baseline_global = eval_global()

        results = []

        with torch.no_grad():

            for t in range(n_tasks):

                print(f"================ TASK {t} ================")

                baseline_task = eval_task(t)
                task_rules = model.get_all_rule_vars()[t]

                for rid in good_rules[t]:

                    rule = task_rules[rid].clone()

                    if rule[10][1] >= 0.5:
                        rule[10] = torch.tensor([0., 0., 1.], device=rule.device)
                        
                    if rule[21][1] >= 0.5:
                        rule[21] = torch.tensor([0., 0., 1.], device=rule.device)
                    
                    success = model.change_rule(t, rid, rule)

                    if not success:
                        print(f"Failed to set red to irrelevant for rule {rid} of task {t}")

                modified_global = eval_global()
                modified_task = eval_task(t)
                
                results.append({
                    "task_id": t,
                    "baseline_global": baseline_global,
                    "modified_global": modified_global,
                    "baseline_task": baseline_task,
                    "modified_task": modified_task,
                    "global_delta": modified_global - baseline_global,
                    "task_delta": modified_task - baseline_task,
                })

            model = MNISTModel.load_from_checkpoint(
                checkpoint_path,
                encoder=MNISTEncoder(
                    emb_size=EMB_SIZE,
                    cp_output=11,
                    number_digits=number_digits
                ),
                rule_module=AECat,
                weights_only=False
            )

            model = model.cpu()

            model.eval()
            model.make_editable()


        pd.DataFrame(results).to_csv("results/mnist_red_bias/corr_rules_red_irr.csv", index=False)


if __name__ == "__main__":

    AECMR_Red().test_red_irr()
    AECMR_Red().test_corr_rules_red_irr()
