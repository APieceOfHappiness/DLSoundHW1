# Automatic Speech Recognition (ASR) with PyTorch


Для запуска данной модели необходимо

```bash
python inference.py datasets.inference.audio_dir=<audio_dir> datasets.inference.transcription_dir=<text_dir> inferencer.from_pretrained=<checkpoint> dataloader.batch_size=32 model.num_layers=5 text_encoder.tokenizer_type='character_wise'
```

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
