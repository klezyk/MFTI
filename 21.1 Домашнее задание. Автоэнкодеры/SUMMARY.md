# Конспект: Автоэнкодеры — AE, VAE, CVAE, Denoising AE, Image Retrieval

## Введение

**Автоэнкодер (autoencoder, AE)** — нейросетевая архитектура вида $x \xrightarrow{\text{encoder}} z \xrightarrow{\text{decoder}} \hat x$, обучаемая воспроизводить вход на выходе. Между энкодером и декодером находится «узкое место» (bottleneck) — низкоразмерное скрытое представление $z \in \mathbb{R}^d$ (latent code, latent embedding). Reconstruction loss $\|x - \hat x\|^2$ или BCE заставляет сеть выжимать из $x$ только самое существенное.

Зачем это нужно:

- **Dimensionality reduction** — латент $z$ занимает в десятки/сотни раз меньше места, чем $x$ (для лиц 45×45×3 = 6075 чисел -> 256-мерный код).
- **Representation learning** — энкодер выучивает «семантически осмысленное» пространство: близкие в латенте картинки похожи и в исходном пространстве.
- **Generative modeling** — декодер позволяет генерировать новые объекты, семплируя $z$ из априорного распределения.
- **Denoising / anomaly detection / image retrieval** — частные случаи, в которых эксплуатируется одно из трёх перечисленных свойств.

В этом конспекте рассмотрены четыре «семейства» автоэнкодеров (Vanilla AE, VAE, CVAE, Denoising AE) и одно практическое применение латентов (image retrieval через KNN). Все четыре модели обучались в рамках задания 21.1 на датасетах LFW (45×45 RGB-лица) и MNIST (28×28 серые цифры).

## 1. Vanilla Autoencoder (AE)

### 1.1 Что это и зачем

Простейший автоэнкодер — детерминированное отображение $x \mapsto \hat x$ через bottleneck. Encoder $E_\phi: \mathbb{R}^{H \times W \times C} \to \mathbb{R}^d$ сжимает изображение, decoder $D_\theta: \mathbb{R}^d \to \mathbb{R}^{H \times W \times C}$ восстанавливает. Loss — попиксельная ошибка реконструкции:

$$\mathcal{L}_{AE}(\phi, \theta) = \mathbb{E}_{x \sim \mathcal{D}} \|x - D_\theta(E_\phi(x))\|_2^2$$

Никаких ограничений на распределение латентов $z = E_\phi(x)$ не накладывается — сеть сама выбирает, как разложить признаки. Это даёт хорошие реконструкции, но **плохое sampling-поведение**: латентное пространство не структурировано (см. раздел 1.4).

### 1.2 Архитектура для изображений

Симметричная свёрточная архитектура с поэтапным понижением разрешения в энкодере (через `Conv2d(stride=2)` или `MaxPool`) и повышением в декодере (через `ConvTranspose2d(stride=2)` или `Upsample` + `Conv2d`):

| Слой | Encoder | Decoder |
|---|---|---|
| 0 | Input 3×45×45 | Linear (latent_dim -> 128·6·6) + Unflatten |
| 1 | Conv 3->32, stride=2 + BN + ReLU | ConvT 128->64, stride=2 + BN + ReLU |
| 2 | Conv 32->64, stride=2 + BN + ReLU | ConvT 64->32, stride=2 + BN + ReLU |
| 3 | Conv 64->128, stride=2 + BN + ReLU | ConvT 32->3, stride=2 + Sigmoid |
| 4 | Flatten + Linear -> latent_dim | Output 3×45×45 |

В решении задания: `latent_dim=256`, 2.55 М параметров, batch=64, 25 эпох на LFW (11828 train / 1315 val).

**Sigmoid** на выходе важна, потому что вход нормализован в [0, 1] (`X / 255.0`) — без сигмоиды декодер мог бы выдавать значения вне этого диапазона.

### 1.3 Loss и обучение

`nn.MSELoss()` + `torch.optim.Adam(lr=1e-3)` — дефолтный набор. На GPU/MPS эпоха LFW 11828 картинок 45×45 занимает 2–3 секунды.

**Финальный результат (задание 21.1):** val MSE = **0.00146** на эпохе 25; train MSE = 0.00132. Лосс монотонно убывал с 0.01574 на первой эпохе до 0.00132 — переобучения нет, gap train/val минимальный.

### 1.4 Главная проблема — несвязное латентное пространство

После обучения попробуем сгенерировать «новые лица», семплируя случайные $z \sim \mathcal{N}(0, I)$ и прогоняя их через decoder. **Получится мусор**: пятна, текстуры без сходства с лицами.

Причина: AE никак не ограничивает распределение латентов. На практике после обучения латенты лиц распределены не как $\mathcal{N}(0, I)$, а с другими mean и std (по каждой координате — свой диапазон). В задании 21.1 mean латента в среднем -0.023, std в среднем 0.643, диапазон по mean от -2.4 до +1.4. Если семплировать из $\mathcal{N}(0, I)$, мы попадаем в «дыру» — область, которую декодер никогда не видел.

**Workaround**: вычислить $\bar\mu, \bar\sigma$ по латентам train-набора, семплировать $z = \bar\mu + \bar\sigma \odot \varepsilon$, $\varepsilon \sim \mathcal{N}(0, I)$. Это даёт уже узнаваемые лица — но всё ещё размытые, потому что латенты разных лиц могут быть мультимодально распределены, а гауссиана с глобальными $\bar\mu, \bar\sigma$ это не учитывает.

Именно эта проблема — мотивация перехода к **VAE**: там KL-член явно стягивает латенты к $\mathcal{N}(0, I)$, и из этого априорного распределения уже можно семплировать напрямую.

### 1.5 Vector arithmetic в латенте (smile vector)

Один из самых известных эффектов: в латенте AE/VAE **линейные операции имеют семантический смысл**. Если взять группу улыбающихся лиц и группу не улыбающихся, можно вычислить:

$$v_{smile} = \frac{1}{|S^+|}\sum_{x \in S^+} E(x) - \frac{1}{|S^-|}\sum_{x \in S^-} E(x)$$

Этот вектор $v_{smile} \in \mathbb{R}^{256}$ — «направление улыбки» в латентном пространстве. Прибавив его к латенту грустного лица $z'' = z + v_{smile}$ и пропустив через decoder, получим то же лицо, но улыбающееся. В задании это работает на 20 улыбающихся + 20 не улыбающихся примерах из train; $\|v_{smile}\|_2 \approx 3.96$.

Объяснение: при обучении сеть **выучивает направления, соответствующие осмысленным атрибутам** (улыбка, очки, борода, цвет волос). Это не магия — просто следствие того, что декодер обязан реализовать в одной непрерывной функции отображение из латента в пиксели всех возможных лиц, и оптимально для него — выделить независимые «оси» под независимые атрибуты.

Аналогичный эксперимент со словами word2vec (`king - man + woman = queen`) — та же история, только в другой модальности.

## 2. Variational Autoencoder (VAE)

### 2.1 Идея

VAE [(Kingma & Welling, 2013)](https://arxiv.org/abs/1312.6114) — **probabilistic autoencoder**: энкодер выдаёт не точку $z$, а распределение $q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x))$ с диагональной ковариацией. Латент $z$ семплируется из этого распределения, потом декодируется.

Зачем это нужно: цель — заставить $q_\phi(z|x)$ быть близким к стандартной гауссиане $p(z) = \mathcal{N}(0, I)$. Тогда после обучения **sampling из $\mathcal{N}(0, I)$ даёт работающие $z$**, потому что обученный декодер видел такие $z$ во время обучения.

### 2.2 ELBO и формула loss

VAE максимизирует **Evidence Lower BOund (ELBO)** на правдоподобие данных:

$$\log p(x) \geq \mathbb{E}_{z \sim q_\phi(z|x)} \log p_\theta(x|z) - D_{KL}(q_\phi(z|x) \| p(z))$$

Минус ELBO — это loss:

$$\mathcal{L}_{VAE} = \underbrace{D_{KL}\bigl(q_\phi(z|x) \| \mathcal{N}(0, I)\bigr)}_{\text{регуляризация}} + \underbrace{\bigl(-\mathbb{E}_{z} \log p_\theta(x|z)\bigr)}_{\text{reconstruction}}$$

**KL для двух гауссиан с диагональной ковариацией** (аналитически):

$$D_{KL}\bigl(\mathcal{N}(\mu, \sigma^2) \| \mathcal{N}(0, I)\bigr) = -\frac{1}{2}\sum_{i=1}^d \bigl(1 + \log\sigma_i^2 - \mu_i^2 - \sigma_i^2\bigr)$$

Если сеть выдаёт `logsigma` $= \log\sigma^2$ (стандартная конвенция), формула в коде:

```python
def KL_divergence(mu, logsigma):
    return -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp())
```

**Reconstruction (Bernoulli decoder для MNIST):** BCE на каждый пиксель:

$$-\log p_\theta(x|z) = -\sum_{ij} \bigl[x_{ij}\log \hat x_{ij} + (1-x_{ij})\log(1-\hat x_{ij})\bigr]$$

В коде: `F.binary_cross_entropy(recon, x, reduction=''sum'')`.

**Важно про `reduction=''sum''`.** Если использовать `''mean''`, KL и BCE получат разный масштаб (KL пропорционален `latent_dim`, BCE — числу пикселей), и KL «задушит» reconstruction. Сумма по всем измерениям даёт сопоставимые слагаемые.

### 2.3 Reparameterization trick

Прямое семплирование $z \sim \mathcal{N}(\mu, \sigma^2)$ непропустит градиент через $\mu$ и $\sigma$ — операция семплирования недифференцируема. Решение: переписать через детерминированную функцию от шума:

$$z = \mu + \sigma \cdot \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)$$

Теперь градиенты по $\mu, \sigma$ текут через эту формулу, а стохастичность изолирована в $\varepsilon$, по которой градиенты не нужны.

```python
def gaussian_sampler(self, mu, logsigma):
    if self.training:
        std = torch.exp(0.5 * logsigma)   # std из log(sigma^2)
        eps = torch.randn_like(std)
        return mu + std * eps
    return mu  # на инференсе берём mode
```

### 2.4 Архитектура

Encoder: свёртки + flatten + два параллельных Linear-слоя для $\mu$ и `logsigma`. Decoder: симметричный, принимает $z$ и выдаёт reconstruction размера входа. В задании 21.1: `latent_dim=32`, conv encoder 1->32->64 + fc bottleneck, conv-transpose decoder, batch=128, 15 эпох на MNIST.

**Финальный результат:** test ELBO/img = **92.13** на эпохе 15 (стартовое 107.7 на эпохе 1). Train/test gap минимален — переобучения нет.

### 2.5 Sampling из $\mathcal{N}(0, I)$ и TSNE

После обучения VAE семплирование `z ~ randn(N, latent_dim)` + `decode(z)` даёт **узнаваемые цифры** (в отличие от AE). Это прямое следствие KL-регуляризации: латенты test-набора плотно покрывают окрестность нуля.

**TSNE визуализация** латентов test (10000 точек, $\mu$ из encoder, perplexity=30, random_state=42) показывает **чёткие кластеры по цифрам 0–9**, хотя VAE обучался без меток. Это значит, что **VAE неявно выучивает классы** — близкие в семантике (4 и 9, 3 и 8) лежат рядом, ортогональные (1 и 0) — далеко.

## 3. Conditional VAE (CVAE)

### 3.1 Идея

Что если хотим **управлять классом** генерируемого объекта? Стандартный VAE этого не умеет: латент кодирует и класс, и стиль вперемешку. **Conditional VAE** [(Sohn et al., 2015)](https://papers.nips.cc/paper/5775-learning-structured-output-representation-using-deep-conditional-generative-models) добавляет в encoder и decoder условие $y$ (метку класса):

$$q_\phi(z|x, y) = \mathcal{N}(\mu_\phi(x, y), \sigma_\phi^2(x, y)), \quad p_\theta(x|z, y)$$

Идея: класс $y$ задаётся **снаружи**, а латент $z$ кодирует только **стиль** (как выглядит этот «3» — толстый или тонкий, наклонный или прямой). Это разделение style/content (или style/class) — главная фишка CVAE.

### 3.2 Архитектура (FC вариант, MNIST)

На MNIST полносвязная сеть работает не хуже свёрточной, проще для CVAE. Условие — one-hot вектор класса `y_oh` $\in \mathbb{R}^{10}$:

| Шаг | Тензор | Размер |
|---|---|---|
| Encoder input | `cat([x_flat, y_oh])` | 784 + 10 |
| Hidden | Linear -> 400 + ReLU | 400 |
| Output | Linear -> $\mu$, Linear -> `logsigma` | latent_dim = 20 |
| Decoder input | `cat([z, y_oh])` | 20 + 10 |
| Hidden | Linear -> 400 + ReLU | 400 |
| Output | Linear -> 784 + Sigmoid | 784 -> reshape (1, 28, 28) |

В задании 21.1: `latent_dim=20`, 20 эпох, batch=128. Финальный test ELBO/img = **91.73** — близко к VAE (92.13), но с возможностью conditional sampling.

### 3.3 Demo: один $z$, разные классы

Главная демонстрация работы CVAE — взять **один** случайный латент $z$ и пропустить его через decoder с разными one-hot метками:

```python
z = torch.randn(1, latent_dim)
for k in range(10):
    y_oh = F.one_hot(torch.tensor([k]), 10).float()
    x_k = decoder(torch.cat([z, y_oh], dim=1))   # цифра k в "стиле z"
```

Результат: десять цифр (0, 1, 2, ..., 9), но все в одном «стиле» — одинаковый наклон, толщина линий, размер. Это подтверждает: $z$ кодирует только variability **внутри** класса, а $y$ задаёт сам класс.

### 3.4 Сравнение TSNE: VAE vs CVAE

- **VAE TSNE**: чёткие кластеры по классам — латент кодирует и класс, и стиль вперемешку.
- **CVAE TSNE**: кластеры **размыты, перекрываются** — латент больше не кодирует класс (тот вынесен в условие), осталась только информация о стиле, общая для всех классов.

Это «успех» CVAE: разделение факторов variability (class — в условии, style — в латенте) приводит к перекрытию кластеров на TSNE.

## 4. Denoising Autoencoder (DAE)

### 4.1 Идея

Простая модификация AE: на вход подаём **зашумлённую** картинку $\tilde x = x + n$, target — чистая $x$. Сеть учится выделять сигнал из шума:

$$\mathcal{L}_{DAE} = \mathbb{E}_{x, n}\|x - D_\theta(E_\phi(x + n))\|_2^2$$

Стандартный шум — гауссовский с `noise_factor=0.5`: `X_noisy = X + 0.5 * np.random.normal(size=X.shape)` с последующим `clip(0, 1)`.

### 4.2 Что выучивает DAE

Чтобы успешно восстановить $x$ из $\tilde x$, сеть **обязана** выделить высокоуровневые признаки (контур, текстуру), потому что попиксельная копия шума бессмысленна. В результате энкодер получается более robust, чем у обычного AE, и латенты лучше работают для downstream-задач (классификация по латентам).

DAE можно рассматривать как **неявный prior** на гладкость манифольда: модель учит «как должна выглядеть правильная картинка» и стягивает зашумлённые точки обратно на манифольд.

### 4.3 Результат задания

Простая FC-архитектура (784 -> 256 -> 64 -> 256 -> 784 + ReLU, Sigmoid на выходе), 12 эпох, batch=128, noise_factor=0.5 на MNIST. Финальный test MSE = **0.01007** (стартовое 0.01687 на эпохе 1). Визуально: noisy input -> чистая реконструкция -> практически совпадает с original.

### 4.4 Применения

- **Image denoising** — astrophotography, microscopy, медицинские снимки.
- **Audio denoising** — на спектрограммах (та же задача в другой модальности).
- **Pretraining encoder''а** — после DAE-обучения energy-based модели хорошо стартуют.
- **Dropout как родственная техника** — внутри сети тоже добавляется шум, но в активациях, а не в input.

## 5. Image Retrieval через AE-латенты

### 5.1 Идея

Encoder обученного AE работает как **feature extractor**: близкие в латенте картинки семантически похожи. Это можно использовать для поиска похожих изображений:

1. Прогнать всю train-выборку через encoder -> получить базу латентов $\{z_i\}$.
2. Построить KNN-индекс на $\{z_i\}$.
3. Для query-картинки $x_q$ получить $z_q$, найти top-K ближайших соседей по евклидову расстоянию.

### 5.2 NearestNeighbors вместо LSHForest

В оригинальном коде задания использовался `sklearn.neighbors.LSHForest` — но он **удалён из sklearn 0.21+** как nondeterministic и плохо поддерживаемый. Современная замена:

```python
from sklearn.neighbors import NearestNeighbors

nn_model = NearestNeighbors(n_neighbors=11, algorithm=''ball_tree'', metric=''euclidean'')
nn_model.fit(latents_train)   # (N, latent_dim)

distances, indices = nn_model.kneighbors(z_query.reshape(1, -1))
similar_images = X_train[indices[0]]
```

`algorithm=''ball_tree''` — exact KNN с метрической структурой, на 256-мерных латентах работает быстро. Для миллионов точек лучше использовать ANN (approximate): FAISS, ScaNN.

### 5.3 Применения retrieval-схемы

- **Face recognition** — энкодер ResNet/FaceNet + KNN на embeddings.
- **Reverse image search** (Google Images, Pinterest) — feature extractor + ANN-индекс.
- **Recommender systems** — продукты как latent vectors, рекомендации = top-K соседи.
- **Content-based image retrieval (CBIR)** в медицине, e-commerce, искусствоведении.

## 6. Сравнительная таблица моделей

| Модель | Loss | Sampling из $\mathcal{N}(0, I)$ | Class control | Кластеры в TSNE | Финальная метрика |
|---|---|---|---|---|---|
| **AE** | MSE | плохо (нужна real-stat) | нет | случайные | val MSE = 0.00146 |
| **VAE** | BCE + KL | работает | нет | по классам | test ELBO/img = 92.13 |
| **CVAE** | BCE + KL | работает + условие | да | размытые/перекрытые | test ELBO/img = 91.73 |
| **DAE** | MSE (noisy -> clean) | n/a (не для генерации) | нет | n/a | test MSE = 0.01007 |

## 7. Ключевые уроки задания

1. **`reduction=''sum''` в VAE-лоссе** обязателен — `''mean''` ломает баланс KL и BCE; KL стягивает всё к нулю, реконструкция превращается в средний пиксель.
2. **Конвенция `logsigma`** — стандартно сеть выдаёт $\log\sigma^2$, тогда std в reparam = `exp(0.5 * logsigma)`. Если выдавать просто $\log\sigma$, формула KL и reparam меняются — путаница в кодах туториалов идёт оттуда.
3. **Sampling из AE НЕ из $\mathcal{N}(0, I)$**, а из реального распределения латентов: `mean + std * randn`. Эта проблема — мотивация VAE.
4. **Vector arithmetic в латенте AE работает** — `mean(smile) - mean(no-smile)` даёт «вектор улыбки» с нормой ~4, который прибавляется к латенту любого лица и делает улыбку.
5. **CVAE concat одинаков в encoder и decoder** — `cat([x, y_oh])` на входе encoder, `cat([z, y_oh])` на входе decoder. Можно конкатенировать на каждом слое (более выразительно), но базовая схема с двумя concat''ами уже даёт хорошее разделение style/class.
6. **TSNE на ~10K точек с perplexity=30, random_state=42** — стандартные настройки. Берётся $\mu$ (mode энкодера), не семпл — для воспроизводимости.
7. **`NearestNeighbors(algorithm=''ball_tree'')` заменяет deprecated `LSHForest`** — последний удалён из sklearn 0.21+.
8. **VAE неявно выучивает классы** — TSNE на латентах MNIST даёт чёткие кластеры по цифрам, хотя обучение шло без меток.
9. **CVAE «убирает» классы из латента** — TSNE на CVAE-латентах показывает размытые перекрывающиеся облака, потому что class теперь живёт в condition, а не в $z$.
10. **Denoising AE — простейший способ выучить robust features** — добавили шум на вход, оставили чистый target, получили энкодер, который выжимает только сигнал.

## 8. Полезные функции

- `torch.nn.Conv2d / nn.ConvTranspose2d` — encoder/decoder свёртки с `stride=2` для понижения/повышения разрешения.
- `torch.nn.functional.binary_cross_entropy(recon, x, reduction=''sum'')` — reconstruction loss для VAE (decoder с Sigmoid на выходе).
- `torch.exp(0.5 * logsigma) * torch.randn_like(mu)` — reparameterization trick (std из конвенции $\log\sigma^2$).
- `torch.nn.functional.one_hot(y_long, num_classes=10).float()` — one-hot encoding для CVAE condition.
- `sklearn.manifold.TSNE(n_components=2, perplexity=30, random_state=42)` — 2D-визуализация латентов.
- `sklearn.neighbors.NearestNeighbors(n_neighbors=11, algorithm=''ball_tree'', metric=''euclidean'')` — KNN для image retrieval.
- `copy.deepcopy(model.state_dict())` — сохранение лучших весов по метрике на валидации.

## 9. Дальнейшее развитие

| Модель | Идея | Когда применять |
|---|---|---|
| **β-VAE** [(Higgins 2017)](https://openreview.net/forum?id=Sy2fzU9gl) | KL домножен на β > 1 — стимулирует disentanglement осей латента | Когда нужны independent factors (face attributes, etc.) |
| **VQ-VAE** [(van den Oord 2017)](https://arxiv.org/abs/1711.00937) | Discrete codebook вместо continuous latent | Препроцессор для autoregressive моделей (DALL·E, MusicLM) |
| **VAE-GAN** [(Larsen 2016)](https://arxiv.org/abs/1512.09300) | Reconstruction loss = perceptual + adversarial вместо MSE/BCE | Высокое визуальное качество (резкие края, текстуры) |
| **Diffusion models** [(Ho 2020)](https://arxiv.org/abs/2006.11239) | Reverse denoising process с тысячами шагов | SOTA генерация (Stable Diffusion, DALL·E 3, Imagen) |
| **StyleGAN3** [(Karras 2021)](https://arxiv.org/abs/2106.12423) | Не VAE, но идейно близко — латент кодирует style | Photorealistic face generation |
| **Autoencoders for anomaly detection** | Reconstruction error как anomaly score | Industrial defect detection, fraud detection |

## Ссылки

- Оригинальная статья VAE: [Kingma & Welling, "Auto-Encoding Variational Bayes", 2013](https://arxiv.org/abs/1312.6114)
- CVAE: [Sohn et al., "Learning Structured Output Representation using Deep Conditional Generative Models", 2015](https://papers.nips.cc/paper/5775-learning-structured-output-representation-using-deep-conditional-generative-models)
- Туториал по VAE (с производной ELBO): [Tutorial on Variational Autoencoders, Doersch 2016](https://arxiv.org/abs/1606.05908)
- Towardsdatascience VAE: [Understanding Variational Autoencoders](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73)
- Denoising AE: [Vincent et al., "Stacked Denoising Autoencoders", 2010](https://www.jmlr.org/papers/v11/vincent10a.html)
- Документация PyTorch: [nn.functional.binary_cross_entropy](https://pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy.html)
- sklearn NearestNeighbors: [User Guide](https://scikit-learn.org/stable/modules/neighbors.html)

---

**Итог.** AE -> VAE -> CVAE — это путь от детерминированного компрессора к контролируемому генератору. Главный шаг — VAE с KL-регуляризацией, который делает латентное пространство пригодным для sampling. CVAE добавляет внешнее условие, отделяя class от style. DAE и retrieval — практические применения того же encoder''а: один учится восстанавливать чистый сигнал из шума, другой — искать похожие объекты по латентам. Все четыре модели за полтора часа обучения на MPS дают вменяемые результаты на стандартных датасетах (LFW и MNIST), и каждая иллюстрирует свою «грань» автоэнкодерной парадигмы.
