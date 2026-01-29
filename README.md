
# synthetic-dataset-utils

Небольшая Python-библиотека для генерации синтетических датасетов и базовой
статистической валидации результатов. Подходит для тестирования моделей,
проверки гипотез и демонстрации навыков работы с данными.

## Возможности
- Генерация линейных регрессионных данных
- Генерация смесей нормальных распределений
- Подсчёт базовых статистик
- Метрики качества (MSE, MAE)
- Сравнение распределений (KS statistic)

## Установка
```bash
pip install synthetic-dataset-utils
```

## Пример использования
```python
from synthetic_dataset_utils import generate_linear_regression, mse

df = generate_linear_regression(n=100, slope=2.5, noise_std=1.2)
error = mse(df["y"], df["y"] * 0.95)
```


