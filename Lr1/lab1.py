# lp_final_FIXED.py
# Исправлено: в фазе 1 суммируем ТОЛЬКО строки с искусственными переменными

import sys
import io

EPS = 1e-10

class SimplexSolver:
    def __init__(self):
        self.n = self.m = 0
        self.c = self.A = self.b = self.signs = None
        self.tableau = None
        self.basis = []
        self.total_vars = 0

    def solve(self):
        # Default data (used if no input file is provided)
        data = [
            "objective max",
            "c 1 5 2 1",
            "1 1 2 0 <= 6",
            "0 1 1 1 = 4",
            "2 0 0 1 >= 3"
        ]
        self.parse(data)
        self.print_task()
        self.build_tableau()
        if not self.phase1():
            print("НЕТ ДОПУСТИМЫХ РЕШЕНИЙ")
            return False
        return self.phase2()

    def solve_file(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f if l.strip()]
        # Allow files starting with BOM or comments starting with #
        lines = [l for l in lines if not l.startswith('#')]
        self.parse(lines)
        self.print_task()
        self.build_tableau()
        if not self.phase1():
            print("НЕТ ДОПУСТИМЫХ РЕШЕНИЙ")
            return False
        return self.phase2()

    def parse(self, lines):
        lines = [l.strip() for l in lines if l.strip()]
        # First line must specify objective (min or max)
        first = lines[0].split()
        if len(first) >= 2 and first[0].lower() == 'objective':
            self.maximize = (first[1].lower() == 'max')
        else:
            self.maximize = True
        self.c = [float(x) for x in lines[1].split()[1:]]
        self.orig_c = list(self.c)
        if not self.maximize:
            # Convert minimization to maximization
            self.c = [-v for v in self.c]
        self.n = len(self.c)
        self.A, self.b, self.signs = [], [], []
        for line in lines[2:]:
            p = line.split()
            sign = p[-2]
            bi = float(p[-1])
            ai = [float(x) for x in p[:-2]]
            self.A.append(ai)
            self.b.append(bi)
            self.signs.append(sign)
        self.m = len(self.A)

    def print_task(self):
        print("Решение задачи:")
        print("Целевая функция: max", self.c)
        for i in range(self.m):
            print(f"Ограничение {i+1}: {self.A[i]} {self.signs[i]} {self.b[i]}")
        print()

    def build_tableau(self):
        slack = sum(1 for s in self.signs if s == "<=")
        surplus = sum(1 for s in self.signs if s == ">=")
        artificial = sum(1 for s in self.signs if s in ("=", ">="))
        self.total_vars = self.n + slack + surplus + artificial

        cols = self.total_vars + 1  # +1 for RHS
        self.tableau = [[0.0] * cols for _ in range(self.m + 1)]

        for i in range(self.m):
            for j in range(self.n):
                self.tableau[i][j] = self.A[i][j]
            self.tableau[i][-1] = self.b[i]

        col = self.n
        self.basis = []
        artificial_rows = []
        self.artificial_cols = []

        for i in range(self.m):
            sign = self.signs[i]
            if sign == "<=":
                self.tableau[i][col] = 1.0
                self.basis.append(col)
                col += 1
            elif sign == ">=":
                self.tableau[i][col] = -1.0    # surplus
                col += 1
                self.tableau[i][col] = 1.0     # artificial
                self.basis.append(col)
                artificial_rows.append(i)
                self.artificial_cols.append(col)
                col += 1
            elif sign == "=":
                self.tableau[i][col] = 1.0
                self.basis.append(col)
                artificial_rows.append(i)
                self.artificial_cols.append(col)
                col += 1
            # No extra identity columns: slack/surplus/artificial vars provide canonical basis

        # ФАЗА 1: ТОЛЬКО СУММА ИСКУССТВЕННЫХ (а не всех строк с artificial)
        # Формируем цель для фазы 1: минимизировать сумму искусственных переменных.
        # В таблице для симплекса целевые коэфф. задаются со знаком минус (для max),
        # поэтому используем вычитание сумм строк с искусственными переменными.
        for i in artificial_rows:
            for j in range(cols):
                self.tableau[self.m][j] -= self.tableau[i][j]

        # Debug prints removed for clarity

    def pivot(self, r, s):
        pivot = self.tableau[r][s]
        for j in range(len(self.tableau[r])):
            self.tableau[r][j] /= pivot
        for i in range(self.m + 1):
            if i == r: continue
            f = self.tableau[i][s]
            for j in range(len(self.tableau[r])):
                self.tableau[i][j] -= f * self.tableau[r][j]
        old = self.basis[r]
        self.basis[r] = s

    def phase1(self):
        print("Фаза 1: решение вспомогательной задачи...")
        while True:
            s = -1
            for j in range(self.total_vars):
                if self.tableau[self.m][j] < -EPS:
                    s = j
                    break
            if s == -1:
                # Нет опорных отрицательных коэффициентов в целевой строке фазы 1
                # Проверим значение целевой функции фазы 1: если оно не 0 -> нет допустимых решений
                if abs(self.tableau[self.m][-1]) > EPS:
                    print("Фаза 1 завершена: минимум искусств. переменных не равен нулю")
                    return False
                print("Фаза 1 завершена успешно")
                return True

            r = -1
            min_r = float('inf')
            for i in range(self.m):
                if self.tableau[i][s] > EPS:
                    ratio = self.tableau[i][-1] / self.tableau[i][s]
                    if ratio < min_r:
                        min_r = ratio
                        r = i
            if r == -1:
                print("Неограниченность (фаза 1)")
                return False
            self.pivot(r, s)

        return abs(self.tableau[self.m][-1]) < EPS

    def phase2(self):
        print("Фаза 2: основная задача...")

        # Обнуляем и ставим -c
        self.tableau[self.m] = [0.0] * len(self.tableau[self.m])
        for j in range(self.n):
            self.tableau[self.m][j] = -self.c[j]

        # Приводим к базису
        for i in range(self.m):
            bc = self.basis[i]
            coef = self.tableau[self.m][bc]
            if abs(coef) > EPS:
                for j in range(len(self.tableau[self.m])):
                    self.tableau[self.m][j] -= coef * self.tableau[i][j]

        while True:
            s = -1
            for j in range(self.total_vars):
                if hasattr(self, 'artificial_cols') and j in self.artificial_cols:
                    # Исключаем искусственные переменные из кандидатов на вход в базис
                    continue
                if self.tableau[self.m][j] < -EPS:
                    s = j
                    break
            if s == -1: break

            r = -1
            min_r = float('inf')
            for i in range(self.m):
                if self.tableau[i][s] > EPS:
                    ratio = self.tableau[i][-1] / self.tableau[i][s]
                    if ratio < min_r:
                        min_r = ratio
                        r = i
            if r == -1:
                print("Неограничена")
                return False
            self.pivot(r, s)

        x = [0.0] * self.n
        for i in range(self.m):
            if self.basis[i] < self.n:
                x[self.basis[i]] = self.tableau[i][-1]
        z = sum(self.c[i] * x[i] for i in range(self.n))
        z_orig = sum(self.orig_c[i] * x[i] for i in range(self.n))

        print("\n" + "="*50)
        for i in range(self.n):
            print(f"x{i+1} = {x[i]:.6f}")
        # Print objective value in original problem orientation
        z_orig = sum(self.orig_c[i] * x[i] for i in range(self.n))
        print(f"Значение целевой функции: {z_orig:.6f}")
        print("="*50)
        self.solution = x
        # Store objective in original problem orientation
        self.z = z_orig
        # End of phase2
        return True

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    solver = SimplexSolver()
    if len(sys.argv) > 1:
        ok = solver.solve_file(sys.argv[1])
    else:
        ok = solver.solve()
    if ok is False:
        sys.exit(1)