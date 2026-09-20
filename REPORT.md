# Отчёт

## Свёрточные сети, перенос обучения, VAE и CLIP

| ФИО | Что сделано |
|-----|-------------|
| Фоменков Макар Никитич | код, эксперименты, отчёт |

### Постановка

Нужно было:

1. обучить CNN с нуля;
2. обучить CNN через transfer learning;
3. обучить GAN или VAE;
4. прогнать image classification через CLIP;
5. сгенерировать картинки VQGAN+CLIP.

Все классификаторы сидят на одном split PetFaces. Генерация — отдельно: CVAE на тех же мордах, VQGAN+CLIP — внешняя модель ImageNet.

### Данные

PetFaces — кропы морды из Oxford-IIIT Pets, ~3211 фото. В этом дампе две папки склеены (`dog_american`, `dog_english`): внутри лежат разные породы. Поэтому класс берётся из **имени файла**, не из директории. Получается 37 пород (12 кошек / 25 собак), split 70/15/15, stratify по породе, seed 13. Split лежит в `splits/` и переиспользуется во всех ноутбуках.

Полные кадры Oxford (`images1`) нужны только как дополнительный тест для CLIP: та же модель, другой кроп.

### 1. CNN с нуля — `01_scratch_cnn.ipynb`

Сеть `FaceIRNet`: inverted residual блоки (depthwise 3×3 + pointwise), SiLU, 96×96. Две головы — порода и вид. Вид идёт как auxiliary loss с весом 0.4.

Обучение: AdamW, cosine, label smoothing, class weights, mixup на breed-голове, RandomErasing, grad clip. Смотрю top-1 / top-3 и accuracy вида и с species-головы, и через маппинг породы.

### 2. Transfer learning — `02_transfer.ipynb`

EfficientNet-B0, ImageNet веса, 224px, imagenet-нормы. Сначала linear probe (замороженный backbone), потом fine-tune двух последних блоков features + classifier, lr ниже. Для контроля — короткий прогон EfficientNet-B0 без претрейна: если без ImageNet проседает, буст не от «просто более жирной архитектуры».

Сравнение со scratch честное: тот же csv-split, те же 37 классов.

### 3. CVAE — `03_cvae.ipynb`

GAN на этом объёме легко схлопывается, поэтому β-CVAE 64×64, латент 64, условие — кот/собака. KL warmup + free bits, чтобы posterior не умер в нуле. Смотрю реконструкцию, сэмплы из prior при фиксированном условии (один и тот же z, разный вид) и линейную интерполяцию между двумя мордами.

### 4. CLIP — `04_clip_zeroshot.ipynb`

ViT-B/32, без дообучения. Классовый эмбеддинг — среднее по пяти текстовым шаблонам. Метрики на test PetFaces и на подвыборке полных кадров Oxford. Сверточные модели тут как точка отсчёта, CLIP специально zero-shot.

### 5. VQGAN+CLIP — `05_vqgan_clip.ipynb`

Чекпоинт `vqgan_models/last.ckpt` (ImageNet, f=16, codebook 16384) + CLIP ViT-B/32. Оптимизирую непрерывный латент до квантизатора: decode → случайные кропы 224 → CLIP cosine к тексту + лёгкий L2 на z. Три промпта с нуля и один с init image (сиба → neon/cyberpunk). `taming-transformers` ноутбук клонирует сам в `third_party/`, дискриминатор из чекпоинта не гружу — для сэмпла он не нужен.

### Как смотреть

Ноутбуки по порядку 01 → 05. Окружение для 01–03 — `.venv` (torch). CLIP и VQGAN удобнее из `vqgan_clip_env`.

Чекпоинты пишутся в `checkpoints/`, картинки генерации — в `runs/`.
