# Лаба: CNN / transfer / CVAE / CLIP / VQGAN

PetFaces (морды) + Oxford-IIIT Pets в `images1`.

Порядок:

1. `01_scratch_cnn.ipynb` — CNN с нуля, multi-task (порода + вид)
2. `02_transfer.ipynb` — EfficientNet-B0, probe потом fine-tune
3. `03_cvae.ipynb` — conditional VAE
4. `04_clip_zeroshot.ipynb` — CLIP без обучения
5. `05_vqgan_clip.ipynb` — генерация VQGAN+CLIP

Общий код датасета: `petsdata.py`. Классы читаются из имени файла, потому что в PetFaces две папки смешивают породы.

## Окружение

Классификация и VAE:

```
.venv
pip install -r requirements.txt
```

CLIP / VQGAN: удобнее ядро из `vqgan_clip_env`. Ноутбук 05 при первом запуске клонирует `taming-transformers` в `third_party/` и сам оптимизирует латент VQGAN под CLIP, без обёртки над `generate.py`.

## Датасеты

PetFaces: http://www.soshnikov.com/permanent/data/petfaces.tar.gz  
Oxford-IIIT Pets: https://www.robots.ox.ac.uk/~vgg/data/pets/

PetFaces — кропы из Oxford-IIIT Pets, CC BY-SA 4.0, копирайт на фото у исходных авторов.
