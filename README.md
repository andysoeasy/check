# check

# **Оглавление**

1. [Настройка рабочей среды](#0-настройка-рабочей-среды)
    - [1.1 Подключение библиотек](#01-подключение-библиотек)
    - [1.2 Линейная регрессия](#12-линейная-регрессия)
    - [1.3 Вспомогательные функции](#13-вспомогательные-функции)

2. [Отбор признаков](#2-отбор-признаков)
    - [2.1 Постановка задачи](#21-постановка-задачи)
    - [2.2 Алгоритмы отбора признаков](#22-алгоритмы-отбора-признаков)
        - [2.2.1 Алгоритм полного перебора (Full Search)](#221-алгоритм-полного-перебора-full-search)
        - [2.2.2 Алгоритм поочередного добавления и удаления признаков (Add-Del)](#222-алгоритм-поочередного-добавления-и-удаления-признаков-add-del)
        - [2.2.3 Поиск в глубину (DFS)](#223-поиск-в-глубину-dfs)
        - [2.2.4 Поиск в ширину (BFS)](#224-поиск-в-ширину-bfs)
        - [2.2.5 Генетический алгоритм](#225-генетический-алгоритм)
        - [2.2.6 Случайный поиск с адаптацией](#226-случайный-поиск-с-адаптацией)

3. [Анализ работы алгоритмов](#3-анализ-работы-алгоритмов)
    - [3.1 Загрузка данных](#31-загрузка-данных)
    - [3.2 Исследование алгоритмов отбора признаков](#32-исследование-алгоритмов-отбор-признаков)

4. [Выводы](#4-выводы)


# **1. Настройка рабочей среды**

## 1.1 Подключение библиотек

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from functools import lru_cache

from sklearn.datasets import load_diabetes as load_data
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler

import plotly.graph_objects as go
import plotly.express as px

import plotly.io as pio
pio.renderers.default = "notebook+vscode+iframe"
```

## 1.2 Линейная регрессия

```python
class RegressionMetric:
    def __init__(self, method = 'mse'):
        self.method = method
        self.methods = {
            'mse': self._mse
        }
        _acceptable_methods = list(self.methods.keys())
        assert self.method in _acceptable_methods, ValueError(f'This method - {self.method} is unresolved. Please try {_acceptable_methods}')
        
    def _mse(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)
    
    def metric(self, y_pred, y_true):
        return self.methods[self.method](y_pred, y_true)

class LinearRegression(RegressionMetric):
    def __init__(self, method = 'mse'):
        super().__init__(method)
        
    def fit(self, X_train, y_train):
        
        f_pinv = np.dot(np.linalg.inv(np.dot(X_train.T, X_train)), X_train.T)
        self.weights = np.dot(f_pinv, y_train)
        
        return self
        
    def predict(self, X_test):
        return np.dot(X_test, self.weights)
    
    def evaluate(self, X_test, y_test):
        return self.metric(
            y_test,
            self.predict(X_test)
        )
```

## 1.3 Вспомогательные функции

```python
def sochetanie(n, k):
    from math import factorial

    a = factorial(n)
    b = factorial(n - k) * factorial(k)

    return a / b


def naive(target_list):

    eng = RegressionMetric(method = 'mse')

    pred = np.random.choice(target_list.shape[0], size = target_list.shape[0], replace = True)
    return eng.metric(
        target_list,
        pred
    )


def quality(X, y, set_indices = None):
    # Hold Out

    if set_indices is not None:
        X = X[:, np.array(list(set_indices))]
    
    n_samples = X.shape[0]
    
    data_indexes = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(data_indexes)

    train_index, test_index = data_indexes[:int(0.8 * n_samples)], data_indexes[int(0.8 * n_samples):]
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    estimator = LinearRegression(method = 'mse')
    estimator.fit(X_train, y_train)

    return estimator.evaluate(X_test, y_test)


@lru_cache(maxsize=None)
def all_combinations(indices_tuple, k = 0):
    
    indices = list(indices_tuple)
    
    if k == 0:
        result = []
        for size in range(1, len(indices) + 1):
            result.extend(
                all_combinations(indices_tuple, size)
            )
        return tuple(result)
    
    if k > len(indices) or k < 0:
        return ()
    
    if k == 1:
        return tuple((i, ) for i in indices)
    
    result = []
    for i in range(len(indices)):
        current = indices[i]
        remaining = tuple(indices[i + 1:])
        for combo in all_combinations(remaining, k - 1):
            result.append((current,) + combo)
    
    return tuple(result)


def data_for_viz(history):
    ks, errors, feature_sets, hover_texts = [], [], [], []
    best_errors, best_feature_sets = [], []

    for k, combos in history.items():
        if len(combos) == 0:
            continue

        for error, features in combos:
            ks.append(k)
            errors.append(error)
            feature_sets.append(features)

            hover_texts.append(
                f'k={k}<br>Ошибка: {error:.4f}<br>Признаки: {features}'
            )

    for k in sorted(history.keys()):
        if len(history[k]) == 0:
            continue
        best_combo = min(history[k], key = lambda x: x[0])
        best_errors.append(best_combo[0])
        best_feature_sets.append(best_combo[1])  

    return {
        'ks': ks,
        'errors': errors,
        'feature_sets': feature_sets,
        'hover_texts': hover_texts,
        'best_errors': best_errors,
        'best_feature_sets': best_feature_sets
    }
```

# **2. Отбор признаков**


## 2.1 Постановка задачи

$F = \{ f_j: X \rightarrow D_j: j = 1,...,n \}$ - множество признаков.

$\mu_J$ - метод обучения, использующий только признаки $J \subseteq F$.

$Q(J) = Q(\mu_J, X^l)$ - выбранный внешний критерий.

$Q(J) \rightarrow \min$ - задача дискретной оптимизации.


## 2.2 Алгоритмы отбора признаков

### 2.2.1 Алгоритм полного перебора (Full Search)

**Вход**: множество $F$, критерий $Q$, параметр $d$;

1. инициализация: $Q^* := Q(\varnothing)$;
2. **для** $j = 1,...,n$, где $j$ - сложность наборов:
    1. $J_j := arg \min_{J: |J| = j} Q(J)$ - найти лучший набор сложности $j$;
    2. **если** $Q(J_j) < Q^*$ **то** $j^* := j$; $Q^* := Q(J_j)$;
    3. **если** $j - j^* \ge d$ **то вернуть** $J_{j^*}$;

**Преимущества:**
- простота реализации;
- гаранитированный результат;
- полный перебор эффективен, когда:
    * информативных признаков не много $j^* \lesssim 5$;
    * всего признаков не много, $n \lesssim 20..100$.

**Недостатки:**
- в остальных случаях очень долго - $O(2^n)$;
- чем больше перебирается вариантов, тем больше переобучение (особенно, если лучшие из вариантов существенно различны и одинаковы плохи);

**Способы устранения:**
- эвристические методы сокращенного перебора.

```python
def full_search_1_0(X, y, d, return_history = False):
    features_length = X.shape[1]
    indexes = tuple(range(features_length))

    best_q = naive(y)
    best_length = 0
    best_comb = None

    all_info = {
        0: [(best_q, 0)]
    }

    for j in tqdm(indexes, desc = 'Перебираю все наборы'):
        errors_combs = []
        combs = []
        all_info[j + 1] = []

        for combination in tqdm(all_combinations(indexes, k = j + 1), desc = f'Работаю с набором сложностью {j + 1}'):
            error = quality(X[:, combination], y)
            
            combs.append(combination)
            errors_combs.append(error)

            all_info[j + 1].append((error, combination))

        best_comb = np.array(combs)[np.argmin(errors_combs)]
        new_q = np.array(errors_combs)[np.argmin(errors_combs)]

        if new_q < best_q:
            best_length = j + 1
            best_q = new_q
        
        if (j + 1) - best_length >= d:
            if return_history:
                return best_comb, all_info
            
            return best_comb
        
    if return_history:
        return best_comb, all_info

    return best_comb
```


### 2.2.2 Алгоритм поочередного добавления и удаления признаков (Add-Del)

Вход: множество $F$, критерий $Q$, параметр $d$.

**повторять**:
1. **пока** $|J_t| \lt n$ добавлять признаки (итерации Add):
    1. $t := t + 1$ - началась следующая итерация;
    2. $f^* := arg \min_{f \in F \setminus J_{t - 1}} Q(J_{t - 1} \cup \{ f\})$; $J_t := J_{t - 1} \cup \{ f^* \}$;
    3. **если** $Q(J_t) < Q^*$ **то** $t^* := t$; $Q^* := Q(J_t)$;
    4. **если** $t - t^* \ge d$ **то прервать цикл**;
2. **пока** $|J_t| > 0$ удалять признаки (итерации Del): 
    1. $t := t + 1$ - началась следующая итерация;
    2. $f^* := arg \min_{f \in J_{t - 1}} Q(J_{t - 1} \setminus \{ f\})$; $J_t := J_{t - 1} \setminus \{ f^* \}$;
    3. **если** $Q(J_t) < Q^*$ **то** $t^* := t$; $Q^* := Q(J_t)$;
    4. **если** $t - t^* \ge d$ **то прервать цикл**;

**пока** значения критерия $Q(J_{t^*})$ уменьшается;

**вернуть** $J_{t^*}$.

Преимущества:
- как правило, лучше, чем Add и Del по отдельности;
- возможны быстрые инкрементные алогритмы.

Недостатки:
- работает долльше, оптимальность не гарантирует.

```python
def add_del(X, y, d, return_history = False):
    n_features = X.shape[1]
    indexes = np.arange(n_features)

    best_comb = np.array([], dtype = int)
    best_q = naive(y)
    t_star = 0
    n_iter = 0

    history_q = [best_q]

    all_info = {
        'add': {
            i: [] for i in range(1, n_features + 1)
        },
        'del': {
            i: [] for i in range(0, n_features + 1)
        }
    }
    all_info['add'][0] = [(best_q, -1)]

    while True:

        while best_comb.shape[0] < n_features:
            n_iter += 1
            print(f'Итерация add\nНомер итерации: {n_iter}')
            
            features_qualities_current = {}

            for feature in np.setdiff1d(indexes, best_comb):
                to_check = np.union1d(best_comb, feature)
                small_q = quality(X[:, to_check], y)
                features_qualities_current[feature] = small_q

                all_info['add'][to_check.shape[0]].append((small_q, to_check))

            best_feature = sorted(features_qualities_current.items(), key = lambda x: x[1])[0]
            best_comb = np.union1d(best_comb, best_feature[0])
            best_comb_q = best_feature[1]

            print(f'Лучший набор: {best_comb}\n')

            if best_comb_q < best_q:
                t_star = n_iter
                best_q = best_comb_q
            if n_iter - t_star > d:
                history_q.append(best_q)
                break
    
        while best_comb.shape[0] > 0:
            n_iter += 1
            print(f'Итерация del\nНомер итерации: {n_iter}')

            features_qualities_current = {}

            for feature in best_comb:
                to_check = np.setdiff1d(best_comb, feature)
                small_q = quality(X[:, to_check], y)
                features_qualities_current[feature] = small_q

                all_info['del'][to_check.shape[0]].append((small_q, to_check))

            best_feature = sorted(features_qualities_current.items(), key = lambda x: x[1])[-1]
            best_comb = np.setdiff1d(best_comb, best_feature[0])
            best_comb_q = best_feature[1]

            print(f'Лучший набор: {best_comb}\n')

            if best_comb_q < best_q:
                t_star = n_iter
                best_q = best_comb_q
            
            if n_iter - t_star > d:
                history_q.append(best_q)
                break
         
        if abs(history_q[-1] - history_q[-2]) <= 1e-8:
            break

    if return_history:
        return best_comb, all_info
    
    return best_comb
```

### 2.2.3 Поиск в глубину (DFS)

Идеи:
1. Нумерация признаков по возрастанию номеров - чтобы избежать повторов при переборе подмножеств.
2. Если набор $J$ бесперспективен, то больше не пытаться его наращивать.

Обозначим $Q_j^*$ - значение критерия на самом лучшем наборе мощности $j$ из всех до сих пор просмотренных.

Оценка перспективности: набор $J$ не нарищивается, если найдется $j$ такой, что:

$$
\begin{cases}
    Q(J) \ge (1 + \delta) Q(^*_j) \\
    |J| \ge j + d
\end{cases}
$$,
где $d \ge 0$ и $\delta \ge 0$ - параметры.

Чем меньше $d$ и $\delta$, тем сильнее сокращается перебор.

__________________________________________

**Вход:** множество $F$, критерий $Q$, параметры $d$ и $\delta$.

**процедура** ***нарастить***($J \subseteq F$)
1. **если** найдется $j: j \le |J| - d$ и $Q(J) \ge (1 + \delta) Q_j^*$ **то** набор $J$ бесперспективный; **выход**;
2. $Q_{|J|}^* := \min\{Q_{|J|}^*, Q(J)\}$;
3. для всех $f_s \in F$ таких, что $s \gt \max\{ t | f_t \in J \}$: нарастить ($J \cup \{ f_s \}$);

инициализировать массиа лучших значений критерия:
$$Q_j^* := Q(\varnothing), \forall j =1,...,n$$

упорядочить признаки по убыванию информативности.

**нарастить**($\varnothing$);

вернуть $J$, для которого $Q(J) = \min_{j=1,...,n} Q(_j^*)$.

__________________________________________

```python
class DepthFirstSearch:

    def __init__(self, d = 3, delta = 0.0):
        self.d = d
        self.delta = delta

    def _is_promising(self, current_q, current_size):
        if current_size < self.d:
            return True

        for j in range(0, current_size - self.d + 1):
            bound = self.best_by_size[j]
            if np.isfinite(bound) and current_q >= (1 + self.delta) * bound:
                return False
        
        return True

    
    def _expand(self, current_comb, start_idx):
        self.n_iter += 1

        current_comb = np.array(current_comb, dtype = int)
        if current_comb.shape[0] == 0:
            current_q = naive(self.y)
        else:
            current_q = quality(self.X, self.y, current_comb)
        
        current_size = current_comb.shape[0]
        
        if current_size not in self.all_info:
            self.all_info[current_size] = []
        
        self.all_info[current_size].append((current_q, current_comb))

        if current_q < self.best_by_size[current_size]:
            self.best_by_size[current_size] = current_q

        if current_q < self.best_q:
            self.best_q = current_q
            self.best_comb = current_comb.copy()

        if not self._is_promising(current_q, current_size):
            return
        
        for feature in range(start_idx, self.n_features):
            self._expand(
                np.append(current_comb, feature),
                feature + 1
            )

    def fit(self, X, y):
        self.X, self.y = X, y
        self.n_features = X.shape[1]

        self.best_comb = np.array([], dtype = int)
        self.best_q = naive(y)

        self.best_by_size = {i: np.inf for i in range(self.n_features + 1)}
        self.best_by_size[0] = self.best_q

        self.all_info = {
            0: [(self.best_q, [-1])]
        }

        self.n_iter = 0

        self._expand(np.array([], dtype = int), 0)

        return self
```

### 2.2.4 Поиск в ширину (BFS)

Усовершенствованный алгоритм Add: на каждой $j$-ой итерации будем строить не один набор, а множество из $B_j$ наборов, называемое $j$-м рядом:
$$
R_j = \{ J_j^1, ..., J_J^{B_j} \}, J_j^b \subseteq F, |J_j^b| = j, b=1,...,B_j
$$,
где $B_j \le B$ - параметр ширины пучка поиска.

_________________________________________________________
**Вход**: множество $F$, критерий $Q$, параметры $d, B$.

первый ряд состоит из всех наборов длины 1:

$
R_1 := \{ \{ f_1 \}, ..., \{ f_n \} \}; Q^* = Q(\varnothing)
$

**для** $j=1,...,n$, где $j$ - сложность наборов:
1. отсортировать ряд $R_j = \{ J_j^1, ..., J_j^{B_j} \}$ по возрастанию критерия: $Q(J_j^1) \le ... \le Q(J_j^{B_j})$;
2. **если** $B_j \gt B$ **то** $R_j := \{ J_j^1, ..., J_j^B \}$ - оставить $B$ лучших наборов ряда;
3. **если** $Q(J_j^1) \lt Q^*$ **то** $j^* := j$; $Q^* := Q(J_j^1)$;
4. **если** $j - j^* \ge d$ **то вернуть** $J_{j^*}^1$;
5. породить следующий ряд: $R_{j+1} := \{ J \cup \{ f \} | J \in R_j, f \in F \setminus J \}$.
___________________________________________________________

**Трудоемкость:** $O(Bn^2)$, точнее $O(Bn(j^* + d))$.

**Проблема дубликатов.** После сортировки $Q(J_j^1) \le ... \le Q(J_j^{B_j})$ проверить на совпадение только соседние наборы с равными значениями внутреннего и внешнего критерия.

**Адаптивный отбор признаков.** На последнем шаге добавлять к $j$-му ряду только признаки $f$ с наибольшей информативностью $I_j(f)$:
$$
I_j(f) = \sum_{b = 1}^{B_j} [f \in J_j^b]
$$

```python
class BreadthFirstSearch:

    def __init__(self, d = 3, B = 5):
        self.d = d
        self.B = B

    def _evaluate(self, combination):
        combination = np.array(combination, dtype = int)
        if combination.shape[0] == 0:
            return naive(self.y)

        return quality(self.X, self.y, combination)

    def _deduplicate_row(self, row):
        unique_row = []
        seen = set()

        for error, combination in row:
            key = tuple(combination)
            if key in seen:
                continue

            seen.add(key)
            unique_row.append((error, np.array(combination, dtype = int)))

        return unique_row

    def fit(self, X, y):
        self.X, self.y = X, y
        self.n_features = X.shape[1]
        self.indexes = np.arange(self.n_features)

        self.best_comb = np.array([], dtype = int)
        self.best_q = naive(y)
        self.best_j = 0
        self.n_iter = 0

        self.all_info = {
            0: [(self.best_q, [-1])]
        }

        current_row = []
        for feature in self.indexes:
            combination = np.array([feature], dtype = int)
            error = self._evaluate(combination)
            current_row.append((error, combination))
            self.n_iter += 1

        for j in range(1, self.n_features + 1):
            if len(current_row) == 0:
                break

            current_row = sorted(current_row, key = lambda x: x[0])
            current_row = self._deduplicate_row(current_row)

            if len(current_row) > self.B:
                current_row = current_row[:self.B]

            self.all_info[j] = [
                (error, combination.copy()) for error, combination in current_row
            ]

            if current_row[0][0] < self.best_q:
                self.best_q = current_row[0][0]
                self.best_comb = current_row[0][1].copy()
                self.best_j = j

            if j - self.best_j >= self.d:
                break

            next_row = []
            for _, combination in current_row:
                last_feature = combination[-1]
                for feature in range(last_feature + 1, self.n_features):
                    if feature in combination:
                        continue

                    new_combination = np.append(combination, feature)
                    error = self._evaluate(new_combination)
                    next_row.append((error, new_combination))
                    self.n_iter += 1

            current_row = next_row

        return self
```

### 2.2.5 Генетический алгоритм

$J \subseteq F$ - индивид.

$R_t := \{ J_t^1, ..., J_t^{B_t} \}$ - поколение.

$\beta = (\beta_j)_{j=1}^n$, $\beta_j = [f_j \in J]$ - хромосома, кодирующая $J$.

Бинарная операция скрещивания (crossover) $\beta = \beta^\prime \times \beta^{\prime\prime}$:
1. Вариант 1: $\beta_j = \rho_j \beta^{\prime}_j + (1 - \rho_j) \beta_j^{\prime\prime}$, $\rho_j \thicksim uni(0, 1)$.
2. Вариант 2: $\beta = (\beta_1^{\prime}, ..., \beta_s^{\prime}, \beta_{s+1}^{\prime\prime}, ..., \beta_n^{\prime\prime})$, $s \thicksim uni(1,...,n)$. Надо задавать "естественное" ранжирование признаков.

Унарная операция мутации $\beta = \thicksim \beta^{\prime}$:
$$
\beta_j = \rho_j(1 - \beta_j^{\prime}) + (1 - \rho_j)\beta_j^{\prime},
\rho_j \thicksim bin(p_m)
$$,
где $p_m$ - параметр вероятности мутации.

________________________________________________
Вход: 
* множество $F$;
* критерий $Q$;
* параметры $d, p_m$;
* размер популяции $B$;
* число поколений $T$.

инициализировать случайную популяцию из $B$ наборов:

$B_1 := B; R_1 := \{ J_1^1, ..., J_1^{B_1} \}; Q^* := Q(\varnothing)$

для $t = 1, ..., T$, где $t$ - номер поколения:
1. ранжирование индивидов: $Q(J_t^1) \le ... \le Q(J_t^{B_t})$;
2. **если** $B_t \gt B$ **то** ***селекция***: $R_t := \{ J_t^1,...,J_t^B \}$;
3. **если** $Q(J_t^1) \lt Q^*$ **то** $t^* := t; Q^* := Q(J_t^1)$;
4. **если** $t - t^* \ge d$ **то вернуть** $J_{t^*}^1$;
5. породить $t+1$-е поколение путем скрещиваний и мутаций: $R_{t+1} := \{ \thicksim (J^{\prime} \times J^{\prime\prime}) | J^{\prime}, J^{\prime\prime} \in R_t \} \cup R_t$.

```python
class GeneticSearch:

    def __init__(self, d = 3, p_m = 0.1, B = 10, T = 25, random_state = 42):
        self.d = d
        self.p_m = p_m
        self.B = B
        self.T = T
        self.random_state = random_state

    def _encode(self, combination):
        chromosome = np.zeros(self.n_features, dtype = int)
        chromosome[np.array(combination, dtype = int)] = 1
        return chromosome

    def _decode(self, chromosome):
        combination = np.where(np.array(chromosome, dtype = int) == 1)[0]
        if combination.shape[0] == 0:
            combination = np.array([self.rng.integers(0, self.n_features)], dtype = int)
        return combination

    def _evaluate(self, chromosome):
        combination = self._decode(chromosome)
        error = quality(self.X, self.y, combination)

        self.all_info[combination.shape[0]].append((error, combination.copy()))
        self.n_iter += 1

        return error, combination

    def _random_chromosome(self):
        chromosome = self.rng.binomial(1, 0.5, size = self.n_features)
        if chromosome.sum() == 0:
            chromosome[self.rng.integers(0, self.n_features)] = 1
        return chromosome.astype(int)

    def _crossover(self, parent_a, parent_b):
        split = self.rng.integers(1, self.n_features)
        child = np.concatenate([parent_a[:split], parent_b[split:]])
        if child.sum() == 0:
            child[self.rng.integers(0, self.n_features)] = 1
        return child.astype(int)

    def _mutate(self, chromosome):
        mutation_mask = self.rng.binomial(1, self.p_m, size = self.n_features).astype(bool)
        mutated = chromosome.copy()
        mutated[mutation_mask] = 1 - mutated[mutation_mask]
        if mutated.sum() == 0:
            mutated[self.rng.integers(0, self.n_features)] = 1
        return mutated.astype(int)

    def _unique_population(self, population):
        unique_population = []
        seen = set()

        for chromosome in population:
            key = tuple(np.array(chromosome, dtype = int))
            if key in seen:
                continue
            seen.add(key)
            unique_population.append(np.array(chromosome, dtype = int))

        return unique_population

    def fit(self, X, y):
        self.X, self.y = X, y
        self.n_features = X.shape[1]
        self.rng = np.random.default_rng(self.random_state)

        self.best_comb = np.array([], dtype = int)
        self.best_q = naive(y)
        self.t_star = 0
        self.n_iter = 0

        self.all_info = {
            i: [] for i in range(self.n_features + 1)
        }
        self.all_info[0] = [(self.best_q, [-1])]

        population = [self._random_chromosome() for _ in range(self.B)]

        for t in range(1, self.T + 1):
            scored_population = []
            for chromosome in self._unique_population(population):
                error, combination = self._evaluate(chromosome)
                scored_population.append((error, chromosome.copy(), combination.copy()))

            if len(scored_population) == 0:
                break

            scored_population = sorted(scored_population, key = lambda x: x[0])
            selected_population = scored_population[:self.B]

            if selected_population[0][0] < self.best_q:
                self.best_q = selected_population[0][0]
                self.best_comb = selected_population[0][2].copy()
                self.t_star = t

            if t - self.t_star >= self.d:
                break

            parents = [chromosome.copy() for _, chromosome, _ in selected_population]
            next_population = parents.copy()

            while len(next_population) < 2 * self.B:
                parent_ids = self.rng.choice(len(parents), size = 2, replace = True)
                child = self._crossover(parents[parent_ids[0]], parents[parent_ids[1]])
                child = self._mutate(child)
                next_population.append(child)

            population = next_population

        return self
```

### 2.2.6 Случайный поиск с адаптацией

Вход: 
* множество $F$;
* критерий $Q$;
* параметр $d$;
* размер популяции $B$;
* число поколений $T$.

равные вероятности признаков: $p_1 = ... = p_n := \frac{1}{n}$.

инициализировать случайную популяцию из $B_1$ наборов:

$
R_1 := \{ J_1^1, ..., J_1^{B_1} \thicksim \{ p_1, ..., p_n \} \}; Q^* := Q(\varnothing)
$

для $t = 1,...,T$, где $t$ - номер поколения;
1. ранжирование индивидов: $Q(J_t^1) \le ... \le Q(J_t^{B_t})$;
2. **если** $B_t \gt B$ **то** ***селекция***: $R_t := \{ J_t^1,...,J_t^B \}$;
3. **если** $Q(J_t^1) \lt Q^*$ **то** $t^* := t; Q^* := Q(J_t^1)$;
4. **если** $t - t^* \ge d$ **то вернуть** $J_{t^*}^1$;
5. увеличить $p_j$ для признаков из лучших наборов;
6. уменьшить $p_j$ для признаков из худших наборов;
7. породить $t+1$-е поколение из $B_t$ наборов: $R_{t+1} := \{ J^1_{t+1},..., J^{B_t}_{t+1} \thicksim \{ p_1,...,p_n \} \} \cup R_t$

```python
class AdaptiveRandomSearch:

    def __init__(self, d = 3, B = 10, T = 25, alpha = 0.1, random_state = 42):
        self.d = d
        self.B = B
        self.T = T
        self.alpha = alpha
        self.random_state = random_state

    def _normalize_probabilities(self):
        self.probabilities = np.clip(self.probabilities, 1e-3, None)
        self.probabilities = self.probabilities / self.probabilities.sum()

    def _sample_chromosome(self):
        chromosome = self.rng.binomial(1, self.probabilities)
        if chromosome.sum() == 0:
            chromosome[self.rng.choice(self.n_features, p = self.probabilities)] = 1
        return chromosome.astype(int)

    def _decode(self, chromosome):
        combination = np.where(np.array(chromosome, dtype = int) == 1)[0]
        if combination.shape[0] == 0:
            combination = np.array([self.rng.choice(self.n_features, p = self.probabilities)], dtype = int)
        return combination

    def _evaluate(self, chromosome):
        combination = self._decode(chromosome)
        error = quality(self.X, self.y, combination)

        self.all_info[combination.shape[0]].append((error, combination.copy()))
        self.n_iter += 1

        return error, combination

    def _unique_population(self, population):
        unique_population = []
        seen = set()

        for chromosome in population:
            key = tuple(np.array(chromosome, dtype = int))
            if key in seen:
                continue
            seen.add(key)
            unique_population.append(np.array(chromosome, dtype = int))

        return unique_population

    def _update_probabilities(self, selected_population):
        if len(selected_population) == 0:
            return

        selected_chromosomes = np.array([chromosome for _, chromosome, _ in selected_population])
        best_half_size = max(1, len(selected_chromosomes) // 2)
        worst_half_size = max(1, len(selected_chromosomes) // 2)

        best_mean = selected_chromosomes[:best_half_size].mean(axis = 0)
        worst_mean = selected_chromosomes[-worst_half_size:].mean(axis = 0)

        self.probabilities = self.probabilities + self.alpha * (best_mean - worst_mean)
        self._normalize_probabilities()

    def fit(self, X, y):
        self.X, self.y = X, y
        self.n_features = X.shape[1]
        self.rng = np.random.default_rng(self.random_state)

        self.best_comb = np.array([], dtype = int)
        self.best_q = naive(y)
        self.t_star = 0
        self.n_iter = 0

        self.all_info = {
            i: [] for i in range(self.n_features + 1)
        }
        self.all_info[0] = [(self.best_q, [-1])]

        self.probabilities = np.ones(self.n_features, dtype = float) / self.n_features
        population = [self._sample_chromosome() for _ in range(self.B)]

        for t in range(1, self.T + 1):
            scored_population = []
            for chromosome in self._unique_population(population):
                error, combination = self._evaluate(chromosome)
                scored_population.append((error, chromosome.copy(), combination.copy()))

            if len(scored_population) == 0:
                break

            scored_population = sorted(scored_population, key = lambda x: x[0])
            selected_population = scored_population[:self.B]

            if selected_population[0][0] < self.best_q:
                self.best_q = selected_population[0][0]
                self.best_comb = selected_population[0][2].copy()
                self.t_star = t

            if t - self.t_star >= self.d:
                break

            self._update_probabilities(selected_population)

            next_population = [chromosome.copy() for _, chromosome, _ in selected_population]
            while len(next_population) < 2 * self.B:
                next_population.append(self._sample_chromosome())

            population = next_population

        return self
```

# **3. Анализ работы алгоритмов**



## 3.1 Загрузка данных



## 3.2 Исследование алгоритмов отбор признаков



# **4. Выводы**

