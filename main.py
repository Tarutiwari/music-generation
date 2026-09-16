
from __future__ import annotations

import argparse
import os
import random
import shutil
import subprocess
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

IN_COLAB = "google.colab" in sys.modules or os.path.isdir("/content")

DEFAULT_MIDI_DIR = "/content/drive/MyDrive/music/solobeeeee" if IN_COLAB else "data/processed"
DEFAULT_OUT_DIR = "/content/drive/MyDrive/music" if IN_COLAB else "outputs"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from music21 import chord, converter, instrument, note, stream

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adamax

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

if IN_COLAB:
    from google.colab import drive

    drive.mount("/content/drive")


class Config:
    """Everything the original notebook hard-coded, in one place."""

    def __init__(self, midi_dir: str, out_dir: str, epochs: int = 50):
        self.midi_dir = midi_dir
        self.out_dir = out_dir
        self.epochs = epochs
        self.weights_path = os.path.join(out_dir, "piano_lstm.weights.h5")
        self.midi_path = os.path.join(out_dir, "Melody_Generated.mid")
        self.wav_path = os.path.join(out_dir, "Melody_Generated.wav")
        self.plot_path = os.path.join(out_dir, "training_history.png")


SPLIT_NAMES = ("train", "validation", "test")


def list_midi_files(root: str) -> list[Path]:
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in {".mid", ".midi"})


def load_pieces(midi_dir: str) -> dict[str, list[list[str]]]:
    """Return {split: [piece_tokens, ...]}.

    `data/prepare_midi.py` creates train/validation/test subfolders. Pieces are
    kept separate so that training windows never straddle two pieces (a window
    spanning a boundary teaches the model transitions that never happened).
    If no such subfolders exist we return one "all" bucket and split
    chronologically later.
    """
    root = Path(midi_dir)
    files = list_midi_files(midi_dir)
    if not files:
        raise SystemExit(
            f"No MIDI files under {midi_dir}.\n"
            "  python data/fetch_kaggle.py --dataset piano-midi-de\n"
            "  python data/prepare_midi.py --input data/raw --output data/processed"
        )

    buckets: dict[str, list[Path]] = {}
    for path in files:
        parts = path.relative_to(root).parts
        split = parts[0] if len(parts) > 1 and parts[0] in SPLIT_NAMES else "all"
        buckets.setdefault(split, []).append(path)

    pieces: dict[str, list[list[str]]] = {}
    for split, paths in buckets.items():
        parsed_pieces = []
        for path in paths:
            try:
                tokens = extract_notes(converter.parse(path))
            except Exception as exc:
                print(f"Skipping {path.name}: {exc}")
                continue
            if tokens:
                parsed_pieces.append(tokens)
        pieces[split] = parsed_pieces
        print(f"{split:<11} {len(parsed_pieces)} pieces, {sum(map(len, parsed_pieces))} tokens")
    return pieces


def extract_notes(parsed_file) -> list[str]:
    """Flatten a parsed score into a list of note/chord tokens.

    Single notes become pitch names ("C#4"). Chords become dot-joined
    *absolute* MIDI note numbers ("60.64.67"), which keeps the octave.
    """
    notes: list[str] = []

    songs = instrument.partitionByInstrument(parsed_file)
    parts = songs.parts if songs is not None and songs.parts else parsed_file.parts

    for part in parts:
        for element in part.recurse():
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append(".".join(str(p.midi) for p in element.pitches))
    return notes


def filter_rare_pieces(pieces: dict, min_count: int, min_len: int = 1):
    """Drop rare tokens across the whole dataset, then drop pieces left too short."""
    counts = Counter(t for ps in pieces.values() for p in ps for t in p)
    rare = {tok for tok, value in counts.items() if value < min_count}
    print(f"Total notes appearing less than {min_count} times: {len(rare)}")

    filtered: dict[str, list[list[str]]] = {}
    for split, parsed_pieces in pieces.items():
        kept = [[t for t in piece if t not in rare] for piece in parsed_pieces]
        filtered[split] = [p for p in kept if len(p) >= min_len]

    kept_counts = Counter(t for ps in filtered.values() for p in ps for t in p)
    print(f"Corpus size after filtering: {sum(kept_counts.values())} tokens")
    return filtered, kept_counts


def plot_frequencies(count_num: Counter, plot_path: str) -> None:
    recurrence = list(count_num.values())
    print(f"Total unique notes: {len(count_num)}")
    print(f"Average recurrence: {sum(recurrence) / len(recurrence):.1f}")
    print(f"Most frequent note appeared: {max(recurrence)} times")
    print(f"Least frequent note appeared: {min(recurrence)} time(s)")

    plt.figure(figsize=(18, 3), facecolor="#97BACB")
    bins = np.arange(0, max(recurrence) + 50, 50)
    plt.hist(recurrence, bins=bins, color="#97BACB")
    plt.axvline(x=100, color="#DBACC1")
    plt.title("Frequency Distribution of Notes in the Corpus")
    plt.xlabel("Frequency of Notes")
    plt.ylabel("Number of Notes")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"Frequency plot saved: {plot_path}")


def build_mappings(corpus: list[str]) -> tuple[dict, dict, int, int]:
    symb = sorted(set(corpus))
    mapping = {c: i for i, c in enumerate(symb)}
    reverse_mapping = {i: c for i, c in enumerate(symb)}
    print(f"Total notes (after filtering): {len(corpus)}")
    print(f"Number of unique notes: {len(symb)}")
    return mapping, reverse_mapping, len(corpus), len(symb)


def empty_arrays(length: int) -> tuple[np.ndarray, np.ndarray]:
    return np.zeros((0, length), dtype=np.float32), np.zeros((0,), dtype=np.int32)


def sequences_from_pieces(pieces: dict, mapping: dict, length: int = 40) -> dict:
    """Sliding windows built inside each piece, so no window crosses a piece."""
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split, parsed_pieces in pieces.items():
        features, targets = [], []
        for piece in parsed_pieces:
            indices = [mapping[t] for t in piece]
            for i in range(len(indices) - length):
                features.append(indices[i : i + length])
                targets.append(indices[i + length])
        out[split] = (
            np.asarray(features, dtype=np.float32).reshape(-1, length),
            np.asarray(targets, dtype=np.int32),
        )
        print(f"{split}: {len(targets)} sequences")
    return out


def split_data(X: np.ndarray, y: np.ndarray):
    """Chronological split -- overlapping windows leak if we shuffle."""
    n = len(X)
    train_end, val_end = int(n * 0.80), int(n * 0.90)
    return (
        X[:train_end], y[:train_end],
        X[train_end:val_end], y[train_end:val_end],
        X[val_end:], y[val_end:],
    )


def build_model(length: int, vocab_size: int) -> Sequential:
    model = Sequential([
        Input(shape=(length, 1)),
        LSTM(256, return_sequences=True),
        Dropout(0.2),
        LSTM(128),
        Dropout(0.2),
        Dense(vocab_size, activation="softmax"),
    ])
    model.compile(
        optimizer=Adamax(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    model.summary()
    return model


def train(model, X_train, y_train, X_val, y_val, epochs: int):
    return model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=128,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        ],
        verbose=2,
    )


def plot_history(history, plot_path: str) -> None:
    history_df = pd.DataFrame(history.history)
    fig = plt.figure(figsize=(15, 4), facecolor="#97BACB")
    fig.suptitle("Learning Plot of Model for Loss")
    pl = sns.lineplot(data=history_df["loss"], color="#444160")
    pl.set(ylabel="Training Loss")
    pl.set(xlabel="Epochs")
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Training plot saved: {plot_path}")


def chords_n_notes(snippet: list) -> stream.Stream:
    melody: list = []
    offset = 0
    for token in snippet:
        text = str(token)
        if "." in text or text.isdigit():
            notes_in_chord = [note.Note(int(j)) for j in text.split(".")]
            element = chord.Chord(notes_in_chord)
        else:
            element = note.Note(text)
        element.offset = offset
        melody.append(element)
        offset += 1
    return stream.Stream(melody)


def melody_generator(model, seed_pool: np.ndarray, reverse_mapping: dict,
                     scale: float, note_count: int, temperature: float = 0.8):
    seed = seed_pool[np.random.randint(len(seed_pool))].copy()
    generated: list[int] = []

    for _ in range(note_count):
        pred = model(seed[np.newaxis, ...], training=False)[0].numpy()

        pred = np.log(pred + 1e-8) / temperature
        exp_preds = np.exp(pred - np.max(pred))
        pred = exp_preds / np.sum(exp_preds)

        index = int(np.random.choice(len(pred), p=pred))
        generated.append(index)

        new_token = np.array([[[index / scale]]], dtype=np.float32)
        seed = np.concatenate([seed[1:], new_token], axis=0)

    music = [reverse_mapping[c] for c in generated]
    return music, chords_n_notes(music)


def write_midi(melody: stream.Stream, midi_path: str) -> None:
    os.makedirs(os.path.dirname(midi_path) or ".", exist_ok=True)
    melody.write("midi", fp=midi_path)
    print(f"MIDI saved: {midi_path}")


def render_wav(midi_path: str, wav_path: str) -> None:
    """timidity is a Debian package; skip gracefully if it is not installed."""
    if shutil.which("timidity") is None:
        print("timidity not found; skipping WAV render (MIDI is still usable).")
        return
    subprocess.run(["timidity", midi_path, "-Ow", "-o", wav_path], check=False)
    print(f"WAV saved: {wav_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train an LSTM on MIDI and generate a melody.")
    parser.add_argument("--midi-dir", default=os.environ.get("MIDI_DIR", DEFAULT_MIDI_DIR))
    parser.add_argument("--out-dir", default=os.environ.get("OUT_DIR", DEFAULT_OUT_DIR))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--length", type=int, default=40)
    parser.add_argument("--notes", type=int, default=200, help="notes to generate")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--min-count", type=int, default=100)
    parser.add_argument("--generate-only", action="store_true",
                        help="load cached weights instead of training")
    args = parser.parse_args(argv)

    cfg = Config(args.midi_dir, args.out_dir, args.epochs)
    os.makedirs(cfg.out_dir, exist_ok=True)

    pieces = load_pieces(cfg.midi_dir)
    pieces, count_num = filter_rare_pieces(
        pieces, args.min_count, min_len=args.length + 1
    )
    if not any(pieces.values()):
        raise SystemExit("No pieces survived filtering -- lower --min-count.")

    all_tokens = [t for ps in pieces.values() for p in ps for t in p]
    print(f"Total notes in corpus: {len(all_tokens)}")
    print(f"First 20 values: {all_tokens[:20]}")
    plot_frequencies(count_num, os.path.join(cfg.out_dir, "note_frequencies.png"))

    mapping, reverse_mapping, L_corpus, L_symb = build_mappings(all_tokens)
    scale = max(1, L_symb - 1)

    sequences = sequences_from_pieces(pieces, mapping, args.length)
    if set(sequences) == {"all"}:
        print("Flat folder (no split subfolders): chronological 80/10/10 split.")
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(*sequences["all"])
    else:
        empty = empty_arrays(args.length)
        X_train, y_train = sequences.get("train", empty)
        X_val, y_val = sequences.get("validation", empty)
        X_test, y_test = sequences.get("test", empty)

    def scaled(arr: np.ndarray) -> np.ndarray:
        return (arr / scale)[..., np.newaxis].astype(np.float32)

    X_train, X_val, X_test = scaled(X_train), scaled(X_val), scaled(X_test)
    if len(X_train) == 0:
        raise SystemExit("Training split is empty -- check the prepared dataset.")
    if len(X_val) == 0:
        print("Validation split empty; reusing the test split for validation.")
        X_val, y_val = X_test, y_test
    X_seed = X_test if len(X_test) else X_train
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Input shape: {X_train.shape}")

    model = build_model(args.length, L_symb)

    if args.generate_only:
        if not os.path.exists(cfg.weights_path):
            raise SystemExit(f"--generate-only needs {cfg.weights_path}")
        model.load_weights(cfg.weights_path)
        print(f"Weights loaded: {cfg.weights_path}")
    else:
        history = train(model, X_train, y_train, X_val, y_val, args.epochs)
        plot_history(history, cfg.plot_path)
        model.save_weights(cfg.weights_path)
        print(f"Weights saved: {cfg.weights_path}")

    music_notes, melody = melody_generator(
        model, X_seed, reverse_mapping, scale, args.notes, args.temperature
    )
    print(f"Generated {len(music_notes)} tokens: {music_notes[:20]} ...")

    write_midi(melody, cfg.midi_path)
    render_wav(cfg.midi_path, cfg.wav_path)

    if IN_COLAB:
        from google.colab import files

        files.download(cfg.midi_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
