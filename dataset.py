import os
os.environ["HF_HOME"] = r"D:\hf_datasets"
os.environ["HF_DATASETS_CACHE"] = r"D:\hf_datasets\datasets"
os.environ["TRANSFORMERS_CACHE"] = r"D:\hf_datasets\models"
os.environ["HUGGINGFACE_HUB_CACHE"] = r"D:\hf_datasets\hub"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
import torchaudio
import soundfile as sf
import io
import os
from torch.utils.data import Dataset
from datasets import load_dataset, Audio
from typing import Tuple, Union, Optional
import glob
import json
from pathlib import Path


class ViMD(Dataset):
    """
    ViMD Dataset class thiết kế để thay thế LIBRISPEECH.
    Trả về tuple: (waveform, sample_rate, transcript)
    """

    def __init__(
        self,
        dataset_name: str = "ViMD",  # Tên dataset trên HF hoặc path cục bộ
        split: str = "train",        # "train", "valid", hoặc "test"
        target_sample_rate: Optional[int] = 16000,
    ) -> None:
        self.dataset_name = dataset_name
        self.split = split
        self.target_sr = target_sample_rate

        # Load dataset từ Hugging Face (giữ nguyên cấu trúc decode=False để xử lý bytes)
        print(f"Loading ViMD dataset split: {split}...")
#-------------------------------------------------------------KHÚC NÀY ĐÃ SỬA---------------------------------------------------        
        # self._dataset = load_dataset(dataset_name, split=split)
        if split == 'train':
            data_dir = r"D:\hf_datasets\hub\datasets--nguyendv02--ViMD_Dataset\snapshots\3a5b30157034e7eadd5c75fae1a820c6f9383398\data"
            train_files = sorted(glob.glob(os.path.join(data_dir, "train-*.parquet")))

            self._dataset = load_dataset(
                "parquet",
                data_files=train_files,
                split="train"
            )
        if split == 'test':
            data_dir = r"D:\hf_datasets\hub\datasets--nguyendv02--ViMD_Dataset\snapshots\3a5b30157034e7eadd5c75fae1a820c6f9383398\data"
            train_files = sorted(glob.glob(os.path.join(data_dir, "test-*.parquet")))

            self._dataset = load_dataset(
                "parquet",
                data_files=train_files,
                split="train"
            )
        if split == 'valid':
            data_dir = r"D:\hf_datasets\hub\datasets--nguyendv02--ViMD_Dataset\snapshots\3a5b30157034e7eadd5c75fae1a820c6f9383398\data"
            train_files = sorted(glob.glob(os.path.join(data_dir, "valid-*.parquet")))

            self._dataset = load_dataset(
                "parquet",
                data_files=train_files,
                split="train"
            )
#-------------------------------------------------------------KHÚC NÀY ĐÃ SỬA--------------------------------------------------
        # Lọc các cột cần thiết để tối ưu bộ nhớ nếu cần
        self._dataset = self._dataset.select_columns(["audio", "text"])
        self._dataset = self._dataset.cast_column("audio", Audio(decode=False))

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, n: int) -> Tuple[torch.Tensor, int, str]:
        """
        Load sample thứ n.
        
        Returns:
            Tuple(waveform, sample_rate, transcript)
        """
        item = self._dataset[n]
        audio_data = item["audio"]
        transcript = item.get("text", "") # Lấy text hoặc chuỗi rỗng nếu không có

        # 1. Xử lý Audio từ bytes (giống WhisperDataHandler)
        audio_bytes = audio_data["bytes"]
        with io.BytesIO(audio_bytes) as f:
            array, sr = sf.read(f, dtype="float32")

        waveform = torch.from_numpy(array)

        # 2. Chuyển về Mono nếu là Stereo
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=-1)
        
        # Đảm bảo shape là [1, samples] để giống format của torchaudio/LibriSpeech
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        # 3. Resample nếu sample_rate khác target_sr
        if self.target_sr and sr != self.target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.target_sr)
            sr = self.target_sr

        return waveform, sr, transcript

# Ví dụ sử dụng:
# dataset = ViMD(dataset_name="path/to/vimd", split="train")
# waveform, sr, text = dataset[0]


class CachedFeatureDataset(Dataset):
    """
    Dataset wrapper that loads pre-extracted spectrogram features stored on disk.
    Expect each sample to be saved as a torch file that includes
    spectrogram, label, input_length, label_length, and text.
    """

    _DTYPE_MAP = {
        "float32": torch.float32,
        "float16": torch.float16,
        "float64": torch.float64,
    }

    def __init__(self, cache_dir: Union[str, Path], split: Optional[str] = "train", dtype: str = "float32"):
        self.cache_root = Path(cache_dir)
        self.split = split
        self.split_dir = self.cache_root / split if split else self.cache_root
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Cached feature directory not found: {self.split_dir}")

        metadata_path = self.split_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            self.samples = metadata.get("samples", metadata)
        else:
            # Fall back to scanning .pt files
            self.samples = [{"file": p.name} for p in sorted(self.split_dir.glob("*.pt"))]

        if not self.samples:
            raise RuntimeError(f"No cached feature files found in {self.split_dir}")

        dtype = dtype.lower()
        if dtype not in self._DTYPE_MAP:
            raise ValueError(f"Unsupported dtype {dtype}, choose from {list(self._DTYPE_MAP.keys())}")
        self.torch_dtype = self._DTYPE_MAP[dtype]

        # Optional length metadata for smart batching
        self.frame_lengths = [
            sample.get("frames") or sample.get("input_length")
            for sample in self.samples
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        sample_meta = self.samples[index]
        file_name = sample_meta.get("file")
        if file_name is None:
            raise KeyError("Each metadata entry must include a 'file' field pointing to the .pt sample.")
        sample_path = self.split_dir / file_name
        data = torch.load(sample_path, map_location="cpu")
        spectrogram = data["spectrogram"].to(self.torch_dtype)
        label = data["label"]
        input_length = data["input_length"]
        label_length = data["label_length"]
        text = data["text"]
        return spectrogram, label, input_length, label_length, text
