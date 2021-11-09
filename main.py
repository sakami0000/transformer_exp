import time
import warnings
from argparse import ArgumentParser
from pathlib import Path
from runpy import run_path

import pandas as pd
from contiguous_params import ContiguousParams
from sklearn.metrics import roc_auc_score
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig

from src.config import Config
from src.data import get_user_sequences, TrainDataset, ValidDataset
from src.evaluate import predict
from src.optim import NoamLR
from src.utils import set_seed, timer, line_notification
from src.validation import virtual_time_split


@line_notification
def main():
    parser = ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    args = parser.parse_args()

    # settings
    config_path = Path(args.config_file)
    config = Config.load(config_path)

    warnings.filterwarnings("ignore")
    set_seed(config.seed)
    start_time = time.time()

    with timer("load data"):
        DATA_DIR = "./input/riiid-test-answer-prediction/"
        usecols = [
            "row_id",
            "timestamp",
            "user_id",
            "content_id",
            "content_type_id",
            "answered_correctly",
            "prior_question_elapsed_time",
        ]
        dtype = {
            "row_id": "int64",
            "timestamp": "int64",
            "user_id": "int32",
            "content_id": "int16",
            "content_type_id": "int8",
            "answered_correctly": "int8",
            "prior_question_elapsed_time": "float32",
        }

        train_df = pd.read_csv(DATA_DIR + "train.csv", usecols=usecols, dtype=dtype)
        question_df = pd.read_csv(
            DATA_DIR + "questions.csv", usecols=["question_id", "part"]
        )

    train_df = train_df[train_df["content_type_id"] == 0].reset_index(drop=True)

    question_df["part"] += 1  # 0: padding id, 1: start id
    train_df["content_id"] += 2  # 0: padding id, 1: start id
    question_df["question_id"] += 2
    train_df = train_df.merge(
        question_df, how="left", left_on="content_id", right_on="question_id"
    )

    with timer("validation split"):
        train_idx, valid_idx, epoch_valid_idx = virtual_time_split(
            train_df,
            valid_size=config.valid_size,
            epoch_valid_size=config.epoch_valid_size,
        )
        valid_y = train_df.iloc[valid_idx]["answered_correctly"].values
        epoch_valid_y = train_df.iloc[epoch_valid_idx]["answered_correctly"].values

    print("-" * 20)
    print(f"train size: {len(train_idx)}")
    print(f"valid size: {len(valid_idx)}")

    with timer("prepare data loader"):
        train_user_seqs = get_user_sequences(train_df.iloc[train_idx])
        valid_user_seqs = get_user_sequences(train_df.iloc[valid_idx])

        train_dataset = TrainDataset(
            train_user_seqs,
            window_size=config.window_size,
            stride_size=config.stride_size,
        )
        valid_dataset = ValidDataset(
            train_df,
            train_user_seqs,
            valid_user_seqs,
            valid_idx,
            window_size=config.window_size,
        )

        train_loader = DataLoader(train_dataset, **config.train_loader_params)
        valid_loader = DataLoader(valid_dataset, **config.valid_loader_params)

        # valid loader for epoch validation
        epoch_valid_user_seqs = get_user_sequences(train_df.iloc[epoch_valid_idx])
        epoch_valid_dataset = ValidDataset(
            train_df,
            train_user_seqs,
            epoch_valid_user_seqs,
            epoch_valid_idx,
            window_size=config.window_size,
        )
        epoch_valid_loader = DataLoader(
            epoch_valid_dataset, **config.valid_loader_params
        )

    with timer("train"):
        model_file_path = Path(f"src/models/{config.model}.py")
        if model_file_path.exists():
            encoder_config = BertConfig(**config.encoder_config)
            decoder_config = BertConfig(**config.decoder_config)

            encoder_config.max_position_embeddings = config.window_size
            decoder_config.max_position_embeddings = config.window_size

            module = run_path(model_file_path)
            model_class = module["SaintEncoderDecoderModel"]
            model = model_class(encoder_config, decoder_config)

        else:
            raise ValueError(f"Unknown model: {config.model}")

        model.to(config.device)
        model.zero_grad()

        parameters = ContiguousParams(model.parameters())
        optimizer = optim.Adam(parameters.contiguous(), **config.optimizer_params)
        scheduler = NoamLR(optimizer, warmup_steps=config.warmup_steps)
        loss_ema = None

        for epoch in range(config.n_epochs):
            epoch_start_time = time.time()
            model.train()

            progress = tqdm(train_loader, desc=f"epoch {epoch + 1}", leave=False)
            for i, (x_batch, w_batch, y_batch) in enumerate(progress):
                y_pred = model(**x_batch.to(config.device).to_dict())
                loss_fn = nn.BCEWithLogitsLoss(weight=w_batch.to(config.device))
                loss = loss_fn(y_pred, y_batch.to(config.device))
                loss.backward()

                if (
                    config.gradient_accumulation_steps is None
                    or (i + 1) % config.gradient_accumulation_steps == 0
                ):
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    parameters.assert_buffer_is_valid()

                loss_ema = (
                    loss_ema * 0.9 + loss.item() * 0.1
                    if loss_ema is not None
                    else loss.item()
                )
                progress.set_postfix(loss=loss_ema)

            valid_preds = predict(model, epoch_valid_loader, device=config.device)
            valid_score = roc_auc_score(epoch_valid_y, valid_preds)

            elapsed_time = time.time() - epoch_start_time
            print(
                f"Epoch {epoch + 1}/{config.n_epochs} \t valid score: {valid_score:.5f} \t time: {elapsed_time / 60:.1f} min"
            )

    with timer("predict"):
        valid_preds = predict(model, valid_loader, device=config.device)
        valid_score = roc_auc_score(valid_y, valid_preds)

    elapsed_time = time.time() - start_time
    print(f"valid score: {valid_score:.5f}")
    print(f"all processes done in {elapsed_time / 60:.1f} min.")


if __name__ == "__main__":
    main()
