# Mixed Affine Constraints (P) в задачах машинного обучения

Проект по статье: https://arxiv.org/pdf/2602.04479

**Цель:** найти ML-постановки, которые можно записать в форме (P) с аффинными связующими ограничениями, и проверить гипотезу:
> можно ли уменьшить/изменить сложность решения задачи только за счёт эквивалентной переформулировки (без изменения самой задачи).

**Руководитель:** Александр Викторович Рогозин (TG: @rogozin_alexander)

---

## Команда

- Илья Хомченко — VK: https://vk.com/art_of_living_xoma, TG: @He_made_in_heaven  
- Наталья Шелегеда — VK: https://vk.com/pcflslghtr, TG: @pcflslghtr

---

## Итерация 1 (sanity-check)

- Реализована toy-задача Horizontal Federated Learning (квадратичная задача консенсуса с аналитическим решением).
- Проверено влияние топологии графа коммуникации (ring / grid / ER / BA / small-world) на скорость сходимости.
- Артефакт: `code/notebooks/01_sanity.ipynb`

---

## Итерация 2 (эквивалентные формулировки одной задачи)

Мы сравниваем **две эквивалентные** записи HFL-задачи и решаем их ADMM:

1) **Shared-variable consensus** (одна общая переменная z, локальные x_i привязаны к z)  
2) **Edge-based consensus** (ограничения x_i = x_j на рёбрах графа)

**Результат:** на одинаковой задаче и одинаковом графе (Ring) скорость сходимости может существенно различаться из-за формы записи.

- Ноутбук: `code/notebooks/02_reformulation_hfl.ipynb`  
- График: `assets/iter2_shared_vs_edge_ring.png`

---

## Как запустить

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
