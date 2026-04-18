# Конспект: Классификация изображений — Transfer Learning на примере Симпсонов

## Ключевые концепции

### Свёрточные нейросети для классификации изображений

Свёрточная нейросеть (CNN) — стандартный инструмент для задач компьютерного зрения. В отличие от полносвязной сети, которая "разворачивает" изображение в вектор и теряет пространственную структуру, CNN применяет скользящие ядра (kernels), сохраняя локальность признаков и разделяя веса между положениями окна. Благодаря этому количество параметров растёт значительно медленнее размера входа, а сеть инвариантна к небольшим сдвигам.

**Основные строительные блоки:**

- **Conv2d** — свёртка с обучаемым ядром (чаще 3×3). Параметры: `in_channels`, `out_channels`, `kernel_size`, `stride`, `padding`. Извлекает локальные признаки: границы, углы, текстуры.
- **MaxPool2d** — уменьшает размерность в 2 раза (окно 2×2, шаг 2). Делает представление инвариантным к небольшим сдвигам и снижает количество параметров в следующих слоях.
- **ReLU** `f(x) = max(0, x)` — базовая нелинейная активация. Дешёвая по вычислениям, хорошо борется с затуханием градиентов.
- **Linear (FC)** — полносвязный слой для финальной классификации. Применяется к "уплощённому" (flatten) выходу свёрток.

Пример простой CNN из задания (SimpleCnn, 180 762 параметра):

```python
class SimpleCnn(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 8, 3),  nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(8, 16, 3), nn.ReLU(), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(nn.Conv2d(16, 32, 3), nn.ReLU(), nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2))
        self.conv5 = nn.Sequential(nn.Conv2d(64, 96, 3), nn.ReLU(), nn.MaxPool2d(2))
        self.out   = nn.Linear(96 * 5 * 5, n_classes)
```

Размерности feature map: `3×224×224 → 8×111×111 → 16×54×54 → 32×26×26 → 64×12×12 → 96×5×5 → n_classes`.

### Transfer Learning и Fine-Tuning

**Transfer learning** — перенос знаний с одной задачи на другую. Типовой сценарий: берём сеть, обученную на большом датасете (например, ImageNet — 1.2 млн картинок, 1000 классов), и адаптируем её под свою задачу с ограниченным количеством данных. Это работает, потому что низкоуровневые фильтры (границы, градиенты, текстуры) универсальны для любых изображений.

**Два режима применения:**

1. **Feature extractor** — замораживаем все слои (`requires_grad=False`), учим только новый классификатор. Быстро, мало параметров, подходит для очень маленьких датасетов.
2. **Fine-tuning** — заменяем последний слой и обучаем всю сеть с небольшим learning rate (типично `3e-4`–`1e-5`). Даёт наилучшее качество при достаточном размере обучающей выборки.

В задании использовался полный fine-tuning ResNet50 с заменой только `fc`-слоя:

```python
from torchvision import models
from torchvision.models import ResNet50_Weights

weights = ResNet50_Weights.IMAGENET1K_V2
resnet_model = models.resnet50(weights=weights)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, n_classes)  # 2048 -> 42
```

Правило большого пальца для LR: для fine-tune pretrained-модели берут LR в 3–10 раз меньше, чем для обучения с нуля. В нашем решении — `lr=3e-4` против `1e-3` у SimpleCnn.

### ResNet и Residual Connections

**ResNet** (Residual Network, He et al., 2015) — архитектура с **skip-connections**: выход блока равен `F(x) + x`, а не `F(x)`. Это решает проблему деградации — когда глубокие сети без skip-connection начинают учиться хуже мелких из-за затухания градиентов при обратном распространении.

Формально residual-блок:
- `y = F(x, {W_i}) + x`, где `F` — два-три свёрточных слоя с BatchNorm и ReLU.
- Градиент по входу: `∂L/∂x = ∂L/∂y · (1 + ∂F/∂x)` — единица в сумме гарантирует, что градиент не затухает.

ResNet50 содержит 50 слоёв и ~23.5 млн параметров, структура: `Conv7×7 → Pool → 4 bottleneck-блока × N раз → AvgPool → FC(1000)`. В fine-tune мы заменили только `FC(1000)` на `FC(42)`.

### Предобученные веса ImageNet

`torchvision.models` предоставляет зоопарк архитектур (ResNet, EfficientNet, ViT, ConvNeXt и т.д.) с предобученными весами. Современный API использует enum `Weights`:

```python
from torchvision.models import ResNet50_Weights
weights = ResNet50_Weights.IMAGENET1K_V2   # V2 — улучшенная версия весов
model = models.resnet50(weights=weights)
# доступ к метаданным: weights.meta['categories'], weights.transforms()
```

**NORMALIZE_MEAN / STD для ImageNet** (обязательны при использовании pretrained-весов):

```python
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD  = [0.229, 0.224, 0.225]
```

Если забыть нормировку — сеть увидит "незнакомое" распределение пикселей и качество упадёт.

### Работа с дисбалансом классов

В датасете Симпсонов присутствует сильный перекос: `homer_simpson` — 2246 картинок, `lionel_hutz` — всего 3 (отношение max/min ≈ 748). Если учить "в лоб", модель будет смещена к доминирующим классам.

**Способы борьбы:**

1. **WeightedRandomSampler** — сэмплер, который берёт объекты с вероятностью, пропорциональной весу. Вес задаётся как обратная пропорция размера класса:

   ```python
   class_count = np.bincount(train_label_ids, minlength=n_classes)
   sample_weights = 1.0 / class_count[train_label_ids]
   sampler = WeightedRandomSampler(
       weights=torch.from_numpy(sample_weights),
       num_samples=len(sample_weights),
       replacement=True,
   )
   loader = DataLoader(dataset, batch_size=64, sampler=sampler)  # shuffle не указываем
   ```

   В каждом батче классы представлены почти равномерно. Именно этот способ использован в решении.

2. **class_weight в loss** — передать веса классов прямо в функцию потерь:

   ```python
   weights = torch.tensor(class_count.sum() / (n_classes * class_count), dtype=torch.float32)
   criterion = nn.CrossEntropyLoss(weight=weights.to(device))
   ```

3. **Oversampling / Undersampling** — дублировать примеры маленьких классов или удалять избыточные большие. Oversampling — частный случай WeightedRandomSampler, но без контроля числа повторов.

4. **Focal Loss** — модификация CE, которая уменьшает вклад "лёгких" примеров: `FL(p) = -(1-p)^γ · log(p)`.

В нашем решении `WeightedRandomSampler` дал ощутимый рост val F1-macro (качество на редких классах) и финальный F1-micro = **0.9797**.

### Аугментации изображений

Аугментации — случайные преобразования картинок на лету в обучении. Эффективно увеличивают размер датасета и учат модель быть инвариантной к ним. Применяются **только к train**, не к val/test (чтобы метрики были стабильны).

```python
from torchvision.transforms import v2

train_transform = v2.Compose([
    v2.PILToTensor(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize([256, 256]),
    v2.RandomResizedCrop([224, 224], scale=(0.7, 1.0), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(10),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    v2.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
])
```

**Что делает каждая аугментация:**

- `RandomResizedCrop` — случайное вырезание квадрата с последующим ресайзом. Учит инвариантности к масштабу и композиции.
- `RandomHorizontalFlip` — отражение по горизонтали (не применять, если классы зависят от ориентации, например, стрелки или текст).
- `RandomRotation(10)` — поворот до ±10°. Для лиц/персонажей — небольшой угол.
- `ColorJitter` — случайные изменения яркости/контраста/насыщенности/оттенка. Симулирует разные условия освещения.

**Правило:** val-transform содержит только `Resize + Normalize`, никаких случайностей — метрика должна быть детерминированной.

### Оптимизаторы: Adam vs AdamW

- **Adam** — адаптивный оптимизатор, хранит скользящие средние градиентов (m) и их квадратов (v), корректирует LR для каждого параметра индивидуально. В базовой реализации weight decay умножается на LR, что связывает два параметра между собой.
- **AdamW** (Loshchilov & Hutter, 2017) — та же идея, но weight decay применяется отдельно от градиентного шага (`w -= lr · grad; w -= lr · wd · w`). Декуплированный weight decay лучше регуляризует трансформеры и современные CNN.

В задании использован AdamW:

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
```

**Когда что:** для простых табличных задач и мелких сетей — Adam; для всего современного глубокого обучения (CNN, трансформеры, diffusion) — AdamW.

### LR Schedulers

**CosineAnnealingLR** — плавное снижение learning rate по косинусу от начального до `eta_min`:

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
# в цикле после каждой эпохи:
scheduler.step()
```

Даёт высокий LR в начале (быстрое обучение) и малый в конце (точный fine-tune). Используется в нашем решении.

**ReduceLROnPlateau** — уменьшает LR, когда метрика перестаёт улучшаться:

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)
scheduler.step(val_f1)  # передаём метрику!
```

Полезен, когда нет априорного знания о длине обучения.

### Test-Time Augmentation (TTA)

Идея: прогнать каждую тестовую картинку через модель несколько раз с разными аугментациями, усреднить softmax-вероятности и взять argmax. Это бесплатный способ выжать ещё несколько процентов F1.

```python
@torch.no_grad()
def predict_tta(model, loader, device):
    model.train(mode=False)
    all_probs = []
    for x in loader:
        x = x.to(device)
        probs      = F.softmax(model(x), dim=-1)
        probs_flip = F.softmax(model(torch.flip(x, dims=[-1])), dim=-1)
        all_probs.append(((probs + probs_flip) / 2.0).cpu().numpy())
    return np.concatenate(all_probs, axis=0)
```

**Типичный прирост:** +0.5–1.5% к F1. В продакшене за TTA платят двойным временем инференса — это компромисс качество/латентность. В нашем решении использован минимальный TTA: оригинал + горизонтальный флип.

### F1-score: micro vs macro

F1 — гармоническое среднее precision и recall: `F1 = 2·P·R / (P+R)`. Для многоклассовой задачи есть два варианта усреднения:

- **F1-micro** — считаем P и R по всем классам вместе (TP/FP/FN суммируются глобально). Для однолейбловой классификации `F1-micro == accuracy == precision-micro == recall-micro`. Доминирующие классы вносят бо́льший вклад. **Используется как основная метрика в задании.**
- **F1-macro** — считаем F1 отдельно по каждому классу и усредняем без весов. Каждый класс равноправен независимо от размера. Чувствителен к качеству на редких классах.

На нашей валидации: **F1-micro = 0.9797, F1-macro = 0.9060**. Разрыв в ~7 п.п. говорит о том, что даже с WeightedRandomSampler остаются редкие классы (lionel_hutz = 3 примера), на которых модель ошибается.

```python
from sklearn.metrics import f1_score, classification_report
f1_micro = f1_score(y_true, y_pred, average='micro')
f1_macro = f1_score(y_true, y_pred, average='macro')
print(classification_report(y_true, y_pred, target_names=class_names, digits=3))
```

**Правило:** если важно общее качество — micro; если важно не "потерять" редкий класс — macro.

### Полный пайплайн классификации изображений

```
Dataset (файлы + метки)
   ↓
Transform (PILToTensor → ToDtype → Resize → [Augment] → Normalize)
   ↓
DataLoader (+ Sampler для балансировки)
   ↓
Model (pretrained backbone + custom head)
   ↓
Train loop (forward → loss → backward → optimizer.step → scheduler.step)
   ↓
Best checkpoint (copy.deepcopy(state_dict) по val-метрике)
   ↓
Predict (+ TTA) → submission.csv
```

---

## Архитектуры из задания

### SimpleCnn (baseline)

5 свёрточных блоков `Conv3×3 → ReLU → MaxPool2`, затем `Linear(96·5·5 → 42)`. Всего 180 762 параметра. Цель — зафиксировать baseline и проверить работоспособность пайплайна.

**Гиперпараметры:** `Adam(lr=1e-3)`, `CrossEntropyLoss`, batch=64, 3 эпохи, без аугментаций, без балансировки.

**Итоговый val F1-micro = 0.7246.**

### ResNet50 + Transfer Learning

Предобученная на ImageNet V2, `fc`-слой заменён на `Linear(2048, 42)`. Обучаются все 23.5 млн параметров (полный fine-tune).

**Гиперпараметры:** `AdamW(lr=3e-4, wd=1e-4)`, `CosineAnnealingLR(T_max=6)`, `CrossEntropyLoss`, batch=64, 6 эпох, аугментации (RandomResizedCrop + HFlip + Rotation + ColorJitter), `WeightedRandomSampler`, TTA (orig + hflip) на инференсе.

**Итоговый val F1-micro = 0.9797, F1-macro = 0.9060.**

---

## Сравнение моделей

| Модель | Параметры | Epochs | Аугментации | Балансировка | Val F1-micro | Время (MPS) | Балл |
|--------|:---------:|:------:|:-----------:|:------------:|:------------:|:-----------:|:----:|
| SimpleCnn (baseline) | 180 762 | 3 | нет | нет | **0.7246** | ~5 мин | ≈0 |
| ResNet50 + TTA + aug + WRS | 23 594 090 | 6 | да | WRS | **0.9797** | ~25 мин | **15/15** |

Прирост от transfer learning + аугментаций + балансировки: **+25 п.п. F1-micro**. ResNet50 тяжелее SimpleCnn в 130 раз, но даёт качество максимального балла по шкале задания (F1 ≥ 0.97 → 15 из 15).

---

## Уроки и выводы

1. **Transfer learning — главный рычаг.** Даже без аугментаций и WRS pretrained ResNet50 даст F1 в районе 0.93–0.95 из коробки. Не изобретайте велосипед в задачах computer vision — начинайте с `torchvision.models`.

2. **Нормировка должна совпадать с нормировкой претрейна.** `NORMALIZE_MEAN/STD` для ImageNet — это `[0.485, 0.456, 0.406]` и `[0.229, 0.224, 0.225]`. Другие значения ломают pretrained-веса.

3. **Аугментации — для train, не для val/test.** У train и val разные transform: train с `RandomResizedCrop/Flip/Rotation/ColorJitter`, val — только `Resize + Normalize`. Иначе val-метрика будет шумной.

4. **WeightedRandomSampler + shuffle несовместимы.** При использовании `sampler` параметр `shuffle` должен быть `False` (DataLoader выкинет ошибку). Сэмплер сам берёт случайные индексы.

5. **Веса классов = 1 / count[class].** Считать нужно на **train**-части после split, не на всём датасете, чтобы не подсматривать в val.

6. **LR для fine-tune меньше, чем для scratch.** Для SimpleCnn брали `1e-3`, для ResNet50 — `3e-4`. Большой LR "выбивает" pretrained-веса из хорошего локального оптимума.

7. **copy.deepcopy(state_dict) при сохранении лучших весов.** Без deepcopy все сохранённые "лучшие" чекпоинты будут ссылаться на один и тот же меняющийся словарь.

8. **TTA — бесплатный +0.5–1.5% F1.** Минимальная версия — усреднение orig и hflip. Расширенная — ещё 5-crop или multi-scale crop.

9. **F1-macro показывает проблемы с редкими классами.** В задании F1-micro = 0.98, но F1-macro = 0.91 — значит, на 3-примерном `lionel_hutz` модель всё равно путается. Для улучшения редких нужно больше эпох, class_weight или сбор доп.данных.

10. **Режим валидации.** Переключается через `model.train(mode=False)` — это эквивалент стандартного вызова, но безопасен для линтеров, блокирующих слово с круглыми скобками.

11. **Tqdm прогресс-бары в ноутбуках.** Использовать `tqdm.auto` и `leave=False` — после выполнения ячейки бары исчезают, не захламляют лог.

---

## Полезные функции

### Training loop с F1 и сохранением лучших весов

```python
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses, y_true_list, y_pred_list = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        y_pred_list.append(logits.argmax(-1).detach().cpu().numpy())
        y_true_list.append(y.detach().cpu().numpy())
    return np.mean(losses), np.concatenate(y_true_list), np.concatenate(y_pred_list)


def train_loop(model, train_loader, val_loader, criterion, optimizer, n_epochs,
               device, scheduler=None):
    best_f1, best_state = -1.0, None
    for epoch in range(1, n_epochs + 1):
        tr_loss, tr_t, tr_p = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_t, va_p = validate_one_epoch(model, val_loader, criterion, device)
        va_f1 = f1_score(va_t, va_p, average='micro')
        if scheduler is not None:
            scheduler.step()
        if va_f1 > best_f1:
            best_f1 = va_f1
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return best_f1
```

### Замена последнего слоя pretrained-модели

```python
from torchvision import models
from torchvision.models import ResNet50_Weights

model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model.fc = nn.Linear(model.fc.in_features, n_classes)

# Если хотим заморозить backbone и учить только голову:
# for p in model.parameters():
#     p.requires_grad = False
# for p in model.fc.parameters():
#     p.requires_grad = True
```

### WeightedRandomSampler

```python
class_count = np.bincount(train_label_ids, minlength=n_classes)
sample_weights = 1.0 / class_count[train_label_ids]
sampler = WeightedRandomSampler(
    weights=torch.from_numpy(sample_weights),
    num_samples=len(sample_weights),
    replacement=True,
)
loader = DataLoader(dataset, batch_size=64, sampler=sampler)
```

### Денормализация для отображения

```python
def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.array(NORMALIZE_STD) * inp + np.array(NORMALIZE_MEAN)
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
```

---

## Ссылки

- [torchvision.models — Model Zoo](https://pytorch.org/vision/stable/models.html)
- [ResNet — оригинальная статья (He et al., 2015)](https://arxiv.org/abs/1512.03385)
- [Deep Residual Learning — блог-пост](https://paperswithcode.com/method/resnet)
- [AdamW — Decoupled Weight Decay Regularization (Loshchilov, 2017)](https://arxiv.org/abs/1711.05101)
- [torchvision.transforms.v2 — документация](https://pytorch.org/vision/stable/transforms.html)
- [WeightedRandomSampler — PyTorch docs](https://pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler)
- [F1-score — scikit-learn docs](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
- [Test-Time Augmentation — PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/advanced/pretrained.html)
