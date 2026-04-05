# Конспект: Свёрточные и полносвязные нейронные сети

## Ключевые концепции

### Нейронная сеть как обобщение линейной модели

Логистическая регрессия — это простейшая нейронная сеть из одного слоя. Она вычисляет линейную комбинацию $z = Wx + b$ и применяет сигмоиду $\sigma(z) = \frac{1}{1+e^{-z}}$ для получения вероятности класса. Если "наслоить" такие преобразования с нелинейными функциями активации между ними, получится многослойная нейросеть (MLP). Без нелинейностей композиция линейных слоёв эквивалентна одному линейному слою — именно активации дают сети способность моделировать сложные зависимости.

### Функции активации

Функция активации применяется поэлементно к выходу линейного слоя и вводит нелинейность. Выбор активации влияет на сходимость, скорость обучения и устойчивость к проблеме затухающих градиентов (vanishing gradients):

- **Identity** $f(x) = x$ — отсутствие нелинейности. Любая глубокая сеть с Identity эквивалентна одному линейному слою.
- **ReLU** $f(x) = \max(0, x)$ — производная равна 1 при $x > 0$ и 0 при $x \leq 0$. Самая популярная активация, но подвержена проблеме "мёртвых нейронов".
- **LeakyReLU** $f(x) = \max(\alpha x, x)$, обычно $\alpha = 0.01$ — решает проблему мёртвых нейронов, сохраняя малый градиент для отрицательных $x$.
- **ELU** $f(x) = x$ при $x > 0$ и $f(x) = \alpha(e^x - 1)$ при $x \leq 0$ — сглаженная версия LeakyReLU, центрирует выходы около нуля.
- **Tanh** $f(x) = \tanh(x)$ — диапазон $(-1, 1)$, симметричен относительно нуля. Страдает от vanishing gradients: при $|x| \gg 0$ производная $1 - \tanh^2(x)$ стремится к 0.

### Свёртка (Convolution)

Операция свёртки применяет к изображению небольшое ядро (kernel, обычно 3x3 или 5x5) методом скользящего окна. Для каждого положения окна вычисляется сумма поэлементных произведений ядра и соответствующего фрагмента изображения. Это позволяет выделять локальные признаки: границы, углы, текстуры. Ключевые параметры:
- **kernel_size** — размер ядра (чаще 3x3).
- **stride** — шаг сдвига окна (чаще 1).
- **padding** — добавление нулей по краям для сохранения размера.
- **in_channels / out_channels** — количество входных и выходных карт признаков (feature maps).

### Max Pooling

Операция уменьшения размерности: окно (например, 2x2) скользит по карте признаков и возвращает максимум из каждого окна. Уменьшает пространственные размеры в 2 раза, делает представление инвариантным к небольшим сдвигам и снижает количество параметров в следующих слоях.

### Свёрточная нейросеть (CNN)

Архитектура, в которой чередуются свёрточные слои, активации и пулинги. В начале сети извлекаются простые признаки (границы), в глубине — более абстрактные (формы, части объектов). После серии свёрток карты признаков "уплощаются" (flatten) и подаются в полносвязные слои для классификации. CNN значительно эффективнее MLP на изображениях благодаря локальности свёрток и разделению весов (weight sharing).

### Backpropagation

Алгоритм вычисления градиентов функции потерь по параметрам сети через цепное правило дифференцирования. PyTorch автоматически строит вычислительный граф при forward pass и автоматически вычисляет градиенты по каждому параметру при вызове `loss.backward()`. Для каждой активации можно реализовать backward вручную: для ReLU — это индикатор $[x > 0]$, для Tanh — $1 - \tanh^2(x)$.

### Train / Validation режимы

PyTorch-модель имеет два режима: `model.train()` и `model.val_mode()` (через `model.eval()`). Они влияют на поведение слоёв вроде Dropout и BatchNorm. На валидации дополнительно оборачивают код в `torch.no_grad()`, чтобы не строить вычислительный граф и не расходовать память на градиенты.

---

## Базовые блоки PyTorch

### nn.Module

Базовый класс всех моделей. Нужно унаследоваться, определить `__init__` (где регистрируются слои/параметры) и `forward` (прямой проход). Обратный проход вычисляется автоматически через autograd.

```python
import torch
from torch import nn

class MyLinear(nn.Module):
    """Линейный слой, реализованный вручную через nn.Parameter."""
    def __init__(self, in_features, out_features):
        super().__init__()
        # nn.Parameter автоматически регистрируется в model.parameters()
        self.weights = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        return x @ self.weights + self.bias
```

### nn.Parameter

Обёртка над тензором, которая говорит PyTorch: "это обучаемый параметр, следи за его градиентами и включай в model.parameters()". Без этой обёртки тензор-атрибут не будет обновляться оптимизатором.

### autograd

Движок автоматического дифференцирования. Каждая операция над тензорами с `requires_grad=True` регистрируется в вычислительном графе. После вызова `loss.backward()` PyTorch проходит по графу в обратном направлении и заполняет `.grad` у всех листовых параметров.

### TensorDataset и DataLoader

`TensorDataset(X, y)` оборачивает тензоры в датасет, а `DataLoader` разбивает его на батчи. Важные параметры: `batch_size`, `shuffle=True` для train и `shuffle=False` для validation.

```python
from torch.utils.data import TensorDataset, DataLoader

train_ds = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=128, shuffle=False)
```

---

## Архитектуры из задания

### MLP для MNIST (Часть 2)

Полносвязная сеть на основе `nn.Sequential`:

```
Flatten (784) -> Linear(784, 128) -> ELU -> Linear(128, 128) -> ELU -> Linear(128, 10)
```

Количество параметров: **118 282**. Сеть принимает изображение 28x28, разворачивает в вектор 784, два скрытых слоя по 128 нейронов с активацией ELU, выходной слой на 10 классов.

### LeNet для MNIST (Часть 3)

Классическая свёрточная архитектура:

```
Conv2d(1->6, 3x3) -> ReLU -> MaxPool(2x2)
    -> Conv2d(6->16, 3x3) -> ReLU -> MaxPool(2x2)
    -> Flatten
    -> Linear(400->120) -> ReLU
    -> Linear(120->84)  -> ReLU
    -> Linear(84->10)
```

Количество параметров: **60 074** (вдвое меньше MLP, но точность выше). Расчёт размера feature map:
- Вход: 28x28, после Conv2d(3x3) без padding → 26x26, после MaxPool(2) → 13x13
- После Conv2d(3x3) → 11x11, после MaxPool(2) → 5x5
- Flatten: 16 каналов × 5 × 5 = 400 признаков

Формула выхода свёртки: $H_{out} = \lfloor (H_{in} + 2p - k) / s \rfloor + 1$.

---

## Результаты

| Модель | Параметры | Valid Accuracy |
|--------|:---------:|:--------------:|
| Логрегрессия (moons) | 3 | 0.8852 |
| MLP Identity (MNIST) | ~118K | 0.9153 |
| MLP ReLU (MNIST) | ~118K | 0.9694 |
| MLP LeakyReLU (MNIST) | ~118K | 0.9722 |
| MLP Tanh (MNIST) | ~118K | 0.9701 |
| **MLP ELU (MNIST)** | ~118K | **0.9756** (лучшая FC) |
| **LeNet CNN (MNIST)** | 60K | **0.9884** (лучшая) |

Наблюдения:
- **ELU победил среди активаций** — сглаженная нелинейность лучше подходит для MLP.
- **Identity сильно хуже остальных** — без нелинейности сеть вырождается в линейную модель.
- **LeNet обошёл все MLP** при вдвое меньшем числе параметров. CNN использует локальность и разделение весов, что критично для изображений.

---

## Уроки и выводы

1. **Нелинейность обязательна.** Identity-сеть даёт 91.5%, любая нелинейная активация — 97%+. Без активаций сеть не учится.

2. **Свёртки выигрывают на изображениях.** LeNet с 60K параметров обходит MLP со 118K на 1.3 п.п. по точности. Для изображений CNN — стандарт.

3. **copy.deepcopy для state_dict.** При сохранении весов по эпохам обязательно использовать `copy.deepcopy(model.state_dict())` — иначе все сохранённые копии будут ссылаться на один и тот же меняющийся словарь.

4. **optimizer.zero_grad() в начале каждого шага.** Градиенты в PyTorch накапливаются по умолчанию, их нужно обнулять перед каждым `loss.backward()`.

5. **Режимы train и validation переключают поведение.** BatchNorm и Dropout ведут себя по-разному в этих режимах. Дополнительно обернуть валидацию в `with torch.no_grad()` для экономии памяти.

6. **BCEWithLogitsLoss против CrossEntropyLoss.** Для бинарной классификации с логитами — `BCEWithLogitsLoss` (включает сигмоиду). Для многоклассовой — `CrossEntropyLoss` (включает softmax). Не применяй сигмоиду/softmax внутри модели, если используешь эти лоссы.

7. **Формула размера feature map.** Для Conv2d: $H_{out} = \lfloor(H_{in} + 2p - k)/s\rfloor + 1$. Проще ошибиться в расчёте размеров, чем в архитектуре — всегда проверяй `print(x.shape)` в forward при отладке.

---

## Полезные функции

### Training loop

```python
def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    for X, y in loader:
        optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

@torch.no_grad()
def validate(model, loader):
    model.eval()
    correct, total = 0, 0
    for X, y in loader:
        preds = model(X).argmax(-1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / total
```

### Ручной backward для ReLU

```python
class MyReLU(nn.Module):
    def forward(self, x):
        return torch.maximum(torch.zeros_like(x), x)

    def backward(self, x):
        # производная ReLU: 1 если x>0, иначе 0
        grads = torch.zeros_like(x)
        grads[x > 0] = 1
        return grads
```

### Применение свёртки к изображению

```python
import torch.nn.functional as F

kernel = torch.tensor([[-1,-1,-1],[-1, 8,-1],[-1,-1,-1]]).float()
kernel = kernel.reshape(1, 1, 3, 3)  # [out_c, in_c, H, W]
img = torch.randn(1, 1, 28, 28)
out = F.conv2d(img, kernel, padding=1)  # padding сохраняет размер
```

---

## Ссылки

- [nn.Module — PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)
- [nn.Conv2d — PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
- [DataLoader — PyTorch](https://docs.pytorch.org/docs/stable/data.html)
- [Autograd mechanics — PyTorch](https://docs.pytorch.org/docs/stable/notes/autograd.html)
- [CS231n: Convolutional Networks](https://cs231n.github.io/convolutional-networks/)
- [LeNet — оригинальная статья (LeCun, 1998)](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)
