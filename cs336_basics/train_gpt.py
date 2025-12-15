import argparse
from functools import partial
import logging
import os
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from cs336_basics.checkpoint import save_checkpoint
from cs336_basics.data import get_batch
from cs336_basics.functions import cross_entropy
from cs336_basics.modules import TransformerLM
from cs336_basics.optimizer import AdamW, learning_rate_schedule

TRAIN_LOG_INTERVAL = 100
NUM_TRAIN_STEP_VALID = 100
NUM_VALID_BATCHES = 16
NUM_CHECKPOINT_INTERVAL = 1000


def get_model(args) -> nn.Module:
    """
    Helper function to get the model based on the arguments
    """
    model = TransformerLM(
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        context_length=args.context_length,
        theta=args.theta,
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
    )
    logging.info(f"Model {model.modules()}")
    logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    return model


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_input_path: str,
    valid_input_path: str,
    batch_size: int,
    context_length: int,
    num_steps: int,
    checkpoint_path: str,
    lr_schedule: Callable,
    device: str = "mps"
) -> None:
    """
    Train the model for one epoch
    """
    # use the memmap to load the input ids for memory efficiency
    train_input_ids = np.memmap(train_input_path, dtype=np.uint16)
    valid_input_ids = np.memmap(valid_input_path, dtype=np.uint16)

    # running loss for training
    loss_train = 0.0
    for step in range(num_steps):
        # update the learning rate at the beginning of each step
        current_lr = lr_schedule(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        optimizer.zero_grad()
        # get the batch of data for training
        x, y = get_batch(train_input_ids, batch_size, context_length, device)

        # model forward pass
        logits = model(x)

        # loss calculation
        loss = cross_entropy(logits, y)
        loss_train += loss.item()

        # start backward
        loss.backward()

        # update parameters
        optimizer.step()

        # loss logging
        # TODO: use wandb for logging
        if (step + 1) % TRAIN_LOG_INTERVAL == 0:
            print(f"Step {step + 1} loss: {loss_train / TRAIN_LOG_INTERVAL}")
            loss_train = 0.0
            print(f"Learning rate: {current_lr}")

        if (step + 1) % NUM_TRAIN_STEP_VALID == 0:
            print(f"Start validation on step {step + 1}")
            valid_loss = 0.0
            model.eval()
            with torch.no_grad():
                for i in range(NUM_VALID_BATCHES):
                    x_valid, y_valid = get_batch(valid_input_ids, batch_size, context_length, device)
                    logits_valid = model(x_valid)
                    valid_loss += cross_entropy(logits_valid, y_valid)
            print(f"Validation loss: {valid_loss.item() / NUM_VALID_BATCHES}")
            model.train()

        if (step + 1) % NUM_CHECKPOINT_INTERVAL == 0:
            save_checkpoint(model, optimizer, step + 1, os.path.join(checkpoint_path, f"checkpoint_{step + 1}.pt"))


def train(args) -> None:
    """
    The entry point of the training process

    - Load the model and construct the optimizer
    - Train the model for one epoch
    - Checkpoint the model and optimizer
    """
    device = torch.device("mps")
    model = get_model(args)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2), eps=1e-8)
    lr_schedule = partial(
        learning_rate_schedule,
        alpha_max=args.alpha_max,
        alpha_min=args.alpha_min,
        t_warmup=args.t_warmup,
        t_cosine=args.t_cosine,
    )

    train_epoch(
        model=model,
        optimizer=optimizer,
        train_input_path=args.train_input_path,
        valid_input_path=args.valid_input_path,
        batch_size=args.batch_size,
        context_length=args.context_length,
        num_steps=args.num_steps,
        lr_schedule=lr_schedule,
        checkpoint_path=args.checkpoint_path,
        device=device
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # training parameters
    parser.add_argument("--train_input_path", type=str, required=True)
    parser.add_argument("--valid_input_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--context_length", type=int, required=True)
    parser.add_argument("--num_steps", type=int, required=True)

    # optimizer parameters
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--weight_decay", type=float, required=True)
    parser.add_argument("--beta1", type=float, required=True)
    parser.add_argument("--beta2", type=float, required=True)
    parser.add_argument("--alpha_max", type=float, required=True)
    parser.add_argument("--alpha_min", type=float, required=True)
    parser.add_argument("--t_warmup", type=int, required=True)
    parser.add_argument("--t_cosine", type=int, required=True)

    # model parameters
    parser.add_argument("--d_model", type=int, required=True)
    parser.add_argument("--num_heads", type=int, required=True)
    parser.add_argument("--d_ff", type=int, required=True)
    parser.add_argument("--theta", type=float, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--num_layers", type=int, required=True)

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path, exist_ok=True)

    train(args)