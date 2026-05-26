from pathlib import Path
from itertools import product
from collections import defaultdict

import torch
import torchvision
import matplotlib.pyplot as plt

from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import TensorDataset, DataLoader

import torch.nn.functional as F
import lightning.pytorch as pl

from lightning.pytorch.callbacks import ModelCheckpoint

from autoencoderCMR_red import (
    MNISTModel,
    MNISTEncoder,
    AECat,
    InputTypes,
    SimilarityTypes,
    SaveBestModelCallbackVal,
    get_accuracy,
)

from pathlib import Path
from itertools import product
from collections import defaultdict

import torch
import torchvision
import matplotlib.pyplot as plt

from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import accuracy_score


# ============================================================
# CONFIG
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

data_root = Path("./data")

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


# ============================================================
# LOAD MNIST
# ============================================================

def get_mnist_data(train: bool) -> MNIST:
    return torchvision.datasets.MNIST(
        root=str(data_root),
        train=train,
        download=True,
        transform=transform
    )


# ============================================================
# COLORIZATION
# ============================================================

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


# ============================================================
# ADDITION DATASET
# ============================================================

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

    # ========================================================
    # CONCEPT VECTORS
    # ========================================================

    # +1 for RED concept
    c = [
        torch.zeros((len(X[0]), digit_limit + 1)).float()
        for _ in range(len(X))
    ]

    for i, ys in enumerate(y):
        for j, yi in enumerate(ys):

            # One-hot digit concept
            c[i][j, yi] = 1.0

    # ========================================================
    # SUM LABELS
    # ========================================================

    y = torch.sum(torch.stack(y, 0), 0)

    # ========================================================
    # APPLY COLOR BIAS
    # ========================================================

    X, red_mask = apply_color_bias(
        X,
        y,
        train=train,
        train_red_prob=train_red_prob,
        test_red_prob=test_red_prob
    )

    # ========================================================
    # ADD RED CONCEPT TO EACH DIGIT
    # ========================================================

    for i in range(len(c)):

        # Last column = RED concept
        c[i][:, -1] = red_mask.float()

    return X, c, y, red_mask


# ============================================================
# LOGIC EXPLANATIONS
# ============================================================

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

# ============================================================
# VISUALIZATION
# ============================================================

def visualize_samples(
    X,
    y,
    red_mask,
    num_digits,
    num_samples=8,
    title="samples.png"
):
    """
    Visualize dataset examples.
    """

    fig, axes = plt.subplots(
        num_samples,
        num_digits,
        figsize=(2 * num_digits, 2 * num_samples)
    )

    if num_samples == 1:
        axes = [axes]

    for i in range(num_samples):

        for d in range(num_digits):

            img = X[d][i]

            # Convert CHW -> HWC
            img = img.permute(1, 2, 0).numpy()

            axes[i][d].imshow(img)
            axes[i][d].axis("off")

            if d == 0:
                axes[i][d].set_title(
                    f"sum={y[i].item()} | red={bool(red_mask[i])}"
                )

    plt.tight_layout()
    plt.savefig(title)
    print(f"\nSaved visualization to: {title}")

    plt.close()


# ============================================================
# DATASET STATISTICS
# ============================================================

def print_statistics(y, red_mask):

    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    total_red = red_mask.sum().item()

    print(f"Total samples: {len(y)}")
    print(f"Red samples:   {total_red}")
    print(f"Percent red:   {100 * total_red / len(y):.2f}%")

    print("\nPer-sum statistics:")

    for s in sorted(torch.unique(y).tolist()):

        mask = (y == s)

        total = mask.sum().item()

        red = red_mask[mask].sum().item()

        pct = 100 * red / total

        print(
            f"Sum={s:2d} | "
            f"total={total:5d} | "
            f"red={red:5d} | "
            f"{pct:6.2f}%"
        )


# ============================================================
# MAIN
# ============================================================

def c_idx_to_name(c_idx, num_digits, digit_limit):

    total_digit_concepts = num_digits * digit_limit

    if c_idx < total_digit_concepts:

        d = c_idx // digit_limit
        v = c_idx % digit_limit

        return f"d{d}_{v}"

    return "RED"


if __name__ == '__main__':

    number_digits = 2

    print("\nCreating symbolic explanations...\n")

    concept_names, class_names, explanations = \
        create_single_digit_addition(number_digits)

    print("Example concepts:")
    print(concept_names)
    print("...")
    print("Last concept:", concept_names[-1])

    # ========================================================
    # TRAIN DATASET
    # ========================================================

    print("\nCreating TRAIN dataset...")

    X_train, c_train, y_train, red_train = addition_dataset_red(
    train=True,
    num_digits=number_digits,
    train_red_prob=TRAIN_PROB,
    test_red_prob=TEST_PROB
    )

    print_statistics(y_train, red_train)

    # ========================================================
    # TEST DATASET
    # ========================================================

    print("\nCreating TEST dataset...")

    X_test, c_test, y_test, red_test = addition_dataset_red(
    train=False,
    num_digits=number_digits,
    train_red_prob=TRAIN_PROB,
    test_red_prob=TEST_PROB
    )

    print_statistics(y_test, red_test)

    # ========================================================
    # VISUALIZE
    # ========================================================

    visualize_samples(
        X_train,
        y_train,
        red_train,
        number_digits,
        num_samples=30,
        title="train_samples.png"
    )

    visualize_samples(
        X_test,
        y_test,
        red_test,
        number_digits,
        num_samples=8,
        title="test_samples.png"
    )

    # ========================================================
    # TEST DATALOADER
    # ========================================================

    print("\nTesting DataLoader...\n")

    dataset = TensorDataset(*X_train, y_train)

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True
    )

    for batch_idx, batch in enumerate(loader):

        X_batch = batch[:-1]
        y_batch = batch[-1]

        print(f"Batch {batch_idx}")

        for i, xb in enumerate(X_batch):
            print(f"Digit {i} tensor shape: {xb.shape}")

        print("Labels shape:", y_batch.shape)

        print("Example labels:")
        print(y_batch[:10])

        break

    # ========================================================
    # TEST CONCEPT STRUCTURE
    # ========================================================

    print("\nTesting concept tensors...\n")

    for digit_idx, concept_tensor in enumerate(c_train):

        print(f"Digit {digit_idx}")
        print("Shape:", concept_tensor.shape)

        # Expected shape: (N, digit_limit + 1)
        assert concept_tensor.shape[1] == 11, \
            "Expected 10 digit concepts + 1 red concept"

    # ----------------------------------------------------
    # Check one-hot encoding
    # ----------------------------------------------------

    digit_part = concept_tensor[:, :-1]

    one_hot_sums = digit_part.sum(dim=1)

    assert torch.all(one_hot_sums == 1), \
        "Digit concepts are not valid one-hot vectors"

    print("✓ One-hot encoding correct")

    # ----------------------------------------------------
    # Check red concept
    # ----------------------------------------------------

    red_column = concept_tensor[:, -1]

    assert torch.all(red_column == red_train.float()), \
        "Red concept column does not match red_mask"

    print("✓ Red concept matches red_mask")

    # ----------------------------------------------------
    # Show examples
    # ----------------------------------------------------

    print("\nExample concept vectors:")

    for i in range(30):

        digit_class = torch.argmax(
            concept_tensor[i, :-1]
        ).item()

        is_red = bool(concept_tensor[i, -1].item())

        print(
            f"Sample {i}: "
            f"digit={digit_class}, "
            f"red={is_red}"
        )

        print(concept_tensor[i])

    print("\n" + "-" * 50)

    print("\nDONE!")

    # ========================================================
    # FORMAT DATA FOR TRAINING
    # ========================================================

    pl.seed_everything(SEED)

    x_train_tensor = torch.stack(X_train, dim=1).float()
    x_test_tensor  = torch.stack(X_test, dim=1).float()
    print("X train size: ", x_train_tensor.shape)
    print("X test size: ", x_test_tensor.shape)

    c_train_tensor = torch.cat(c_train, dim=-1).float()
    c_test_tensor  = torch.cat(c_test, dim=-1).float()
    print("C train size: ", c_train_tensor.shape)
    print("C test size: ", c_test_tensor.shape)

    y_train_tensor = F.one_hot(y_train.long()).float()
    y_test_tensor  = F.one_hot(y_test.long()).float()
    print("Y train size: ", y_train_tensor.shape)
    print("Y test size: ", y_test_tensor.shape)

    # ========================================================
    # TRAIN / VAL SPLIT
    # ========================================================

    split = int(len(x_train_tensor) * (1 - VAL_SPLIT))

    x_val = x_train_tensor[split:]
    c_val = c_train_tensor[split:]
    y_val = y_train_tensor[split:]

    x_train_final = x_train_tensor[:split]
    c_train_final = c_train_tensor[:split]
    y_train_final = y_train_tensor[:split]

    print("x train: ", x_train_final.shape)
    print("c train: ", c_train_final.shape)
    print("y train: ", y_train_final.shape)

    # ========================================================
    # TRAINING DATALOADERS
    # ========================================================

    train_loader = DataLoader(
        TensorDataset(
            x_train_final,
            c_train_final,
            y_train_final
        ),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(
            x_val,
            c_val,
            y_val
        ),
        batch_size=BATCH_SIZE
    )

    test_loader = DataLoader(
        TensorDataset(
            x_test_tensor,
            c_test_tensor,
            y_test_tensor
        ),
        batch_size=BATCH_SIZE
    )

    # ========================================================
    # MODEL DIMENSIONS
    # ========================================================
    
    n_concepts = c_train_tensor.shape[1]
    n_tasks = y_train_tensor.shape[1]

    print("\nTraining model...")
    print("n_concepts:", n_concepts)
    print("n_tasks:", n_tasks)

    # ========================================================
    # MODEL
    # ========================================================

    model = MNISTModel(

        encoder=MNISTEncoder(
            emb_size=EMB_SIZE,
            cp_output=11,
            number_digits=number_digits
        ),

        emb_size=EMB_SIZE,

        rule_emb_size=RULE_EMB,

        n_tasks=n_tasks,
        n_rules=N_RULES,
        n_concepts=n_concepts,

        concept_names=[
            c_idx_to_name(i, number_digits, 10)
            for i in range(n_concepts)
        ],

        rule_module=AECat,

        lr=LR,

        selector_input=InputTypes.embedding,

        selector_similarity=SimilarityTypes.cosine,

        temperature=0.1,

        reset_selector=True,
        reset_selector_every_n_epochs=40,

        w_c=1,
        w_y=1,
        w_yF=1,

        w_embedding_ae_loss=0.5,
        w_rule_ae_loss=8
    )

    # ========================================================
    # CALLBACKS
    # ========================================================

    cb = SaveBestModelCallbackVal()

    checkpoint = ModelCheckpoint(
        dirpath="./results/mnist_red_bias/",
        filename=f"best_{TRAIN_PROB}_{TEST_PROB}_final",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    # ========================================================
    # TRAINER
    # ========================================================

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        callbacks=[cb, checkpoint],
        deterministic=True,
        gradient_clip_val=0.5,
        gradient_clip_algorithm="norm"
    )

    

    # ========================================================
    # TRAIN
    # ========================================================

    trainer.fit(model, train_loader, val_loader)

    # ========================================================
    # RESTORE BEST MODEL
    # ========================================================

    model.load_state_dict(cb.best_state_dict)

    model.eval()
    model.to("cpu")

    # ========================================================
    # EVALUATION
    # ========================================================

    train_acc = get_accuracy(model, train_loader)
    test_acc  = get_accuracy(model, test_loader)

    print(f"\nTrain Acc: {train_acc:.4f}")
    print(f"Test Acc:  {test_acc:.4f}")

    # ========================================================
    # RULES
    # ========================================================

    print("\nExtracted Rules:\n")

    task_to_rules, _ = model.aggregate_rules(
        train_loader,
        type='most_likely'
    )

    for task in sorted(task_to_rules.keys()):

        print(f"Task {task}:")
        print(task_to_rules[task])
        print()

    print("Train accuracy; ", get_accuracy(model, train_loader))
    print("Test accuracy: ", get_accuracy(model, test_loader))

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
    
    with torch.no_grad():
        for i in range(n_tasks):
            print(f"Task {i} accuracy: ", eval_task(i))

    