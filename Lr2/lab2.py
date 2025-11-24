
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import time
from typing import Callable, Tuple
import warnings
import sys
import io
warnings.filterwarnings('ignore')

# настройка отображения графиков
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class GlobalOptimizer:
    """
    класс для поиска глобального минимума одномерной липшицевой функции
    методом пиявского (метод ломаных)
    """

    def __init__(self, func_str: str, a: float, b: float, eps: float = 0.01):
        """
        инициализация оптимизатора

        параметры:
        func_str : str - строковое представление функции (например, 'x + sin(3.14159*x)')
        a : float - левая граница отрезка
        b : float - правая граница отрезка
        eps : float - точность вычисления
        """
        self.func_str = func_str
        self.a = a
        self.b = b
        self.eps = eps

        # создаем функцию из строки
        self.func = self._create_function(func_str)

        # история точек для визуализации
        self.history = []
        self.iterations = 0
        self.computation_time = 0
        self.func_evals = 0

    def _create_function(self, func_str: str) -> Callable:
        """создает функцию из строкового представления"""
        def f(x):
            namespace = {
                'x': x,
                'sin': np.sin,
                'cos': np.cos,
                'tan': np.tan,
                'exp': np.exp,
                'log': np.log,
                'sqrt': np.sqrt,
                'abs': np.abs,
                'pi': np.pi,
                'e': np.e,
                'np': np
            }
            return eval(func_str, namespace)

        return f

    def _estimate_lipschitz_constant(self, n_samples: int = 1000) -> float:
        """оценка константы липшица методом конечных разностей"""
        x_samples = np.linspace(self.a, self.b, n_samples)
        y_samples = np.array([self.func(x) for x in x_samples])
        # count these evaluations
        try:
            self.func_evals += n_samples
        except Exception:
            pass

        # use central differences if possible for better estimate
        dx = np.diff(x_samples)
        dydx = np.abs(np.diff(y_samples) / dx)
        # pad with endpoints
        differences = dydx
        L = np.max(differences) if differences.size > 0 else 1.0
        # use a safety factor to avoid underestimation
        L = max(L * 1.2, 1e-6)

        return L

    def optimize(self) -> Tuple[float, float]:
        """
        поиск глобального минимума методом пиявского

        возвращает:
        x_min : float - аргумент минимума
        f_min : float - значение минимума
        """
        start_time = time.time()

        # оценка константы липшица
        L = self._estimate_lipschitz_constant()
        self.L = L

        # начальные точки
        fa = self.func(self.a)
        fb = self.func(self.b)
        self.func_evals += 2
        points = [(self.a, fa), (self.b, fb)]
        self.history.append(points.copy())

        self.iterations = 0
        max_iterations = 10000

        self.final_interval = None
        while self.iterations < max_iterations:
            self.iterations += 1

            # сортируем точки по x
            points.sort(key=lambda p: p[0])

            # находим интервал с максимальной характеристикой в Piyavsky
            # характеристика R_i = (f_i + f_j)/2 - L*(x_j - x_i)/2
            max_R = -np.inf
            best_interval_idx = 0

            for i in range(len(points) - 1):
                x_i, f_i = points[i]
                x_j, f_j = points[i + 1]

                delta_x = x_j - x_i
                R = (f_i + f_j) / 2.0 - L * delta_x / 2.0

                if R > max_R:
                    max_R = R
                    best_interval_idx = i

            # получаем лучший интервал
            x_i, f_i = points[best_interval_idx]
            x_j, f_j = points[best_interval_idx + 1]

            # проверяем условие останова: ширина интервала уже меньше eps
            if x_j - x_i < self.eps:
                self.final_interval = (x_i, x_j, f_i, f_j)
                break

            # новая точка: x* = (x_i + x_j)/2 + (f_i - f_j)/(2L)
            # убеждаемся, что новая точка остается в интервале
            x_new = (x_i + x_j) / 2.0 + (f_i - f_j) / (2.0 * L)
            # защита от выхода за границы из-за численной ошибки
            x_new = max(min(x_new, x_j - 1e-12), x_i + 1e-12)
            f_new = self.func(x_new)
            self.func_evals += 1

            points.append((x_new, f_new))

            if self.iterations % 10 == 0 or self.iterations < 20:
                self.history.append(points.copy())

        self.history.append(points.copy())

        # находим минимум
        points.sort(key=lambda p: p[1])
        x_min, f_min = points[0]

        self.computation_time = time.time() - start_time

        return x_min, f_min

    def _lower_envelope(self, x_vals: np.ndarray, points: list) -> np.ndarray:
        """Compute Lipschitz-derived lower envelope (piecewise linear) given sample points and L.
        Returns array of envelope values for x_vals."""
        if not hasattr(self, 'L'):
            L = self._estimate_lipschitz_constant()
        else:
            L = self.L
        # points: list of (xi, fi)
        xi = np.array([p[0] for p in points])
        fi = np.array([p[1] for p in points])
        # For each x, compute max(fi - L * |x - xi|)
        X = x_vals.reshape((-1, 1))  # shape (n_x, 1)
        # compute |X - xi| shaped (n_x, n_points)
        D = np.abs(X - xi.reshape((1, -1)))
        G = fi.reshape((1, -1)) - L * D
        envelope = np.max(G, axis=1)
        return envelope

    def _envelope_vertices(self, points: list, n_grid: int = 2000):
        """Compute exact envelope vertex coordinates (x,y) by detecting argmax changes on fine grid
        and computing intersections between dominating piecewise linear functions."""
        if not hasattr(self, 'L'):
            L = self._estimate_lipschitz_constant()
        else:
            L = self.L
        pts = sorted(points, key=lambda p: p[0])
        xi = np.array([p[0] for p in pts])
        fi = np.array([p[1] for p in pts])
        x_plot = np.linspace(self.a, self.b, n_grid)
        # compute G matrix (n_grid, n_points)
        X = x_plot.reshape((-1, 1))
        D = np.abs(X - xi.reshape((1, -1)))
        G = fi.reshape((1, -1)) - L * D
        arg = np.argmax(G, axis=1)

        seg_changes = [0]
        for k in range(len(arg) - 1):
            if arg[k] != arg[k + 1]:
                seg_changes.append(k + 1)
        seg_changes.append(len(arg))

        # Build vertex list
        vertices_x = []
        vertices_y = []
        for idx in range(len(seg_changes) - 1):
            left = seg_changes[idx]
            right = seg_changes[idx + 1] - 1
            i_dom = arg[left]
            # compute x range for this dominant
            x0 = x_plot[left]
            x1 = x_plot[right]
            # append segment endpoints (we will later compute intersections)
            vertices_x.append(x0)
            vertices_y.append(G[left, i_dom])
            # add right endpoint if it's last segment
            if idx == len(seg_changes) - 2:
                vertices_x.append(x1)
                vertices_y.append(G[right, i_dom])

        # Now compute accurate intersection points between consecutive dominating indices
        accurate_x = [self.a]
        accurate_y = [self.func(self.a) - L * abs(self.a - xi[np.argmax(fi - L * abs(self.a - xi))])]
        for k in range(len(arg) - 1):
            if arg[k] != arg[k + 1]:
                p = arg[k]
                q = arg[k + 1]
                # compute intersection between gp and gq
                xi_p = xi[p]; fi_p = fi[p]
                xi_q = xi[q]; fi_q = fi[q]
                # slope for p at midpoint
                xm = x_plot[k]
                slope_p = L if xm <= xi_p else -L
                slope_q = L if xm <= xi_q else -L
                intercept_p = fi_p - slope_p * xi_p
                intercept_q = fi_q - slope_q * xi_q
                denom = slope_p - slope_q
                if abs(denom) < 1e-12:
                    x_inter = x_plot[k]
                else:
                    x_inter = (intercept_q - intercept_p) / denom
                # clip into [a,b]
                x_inter = max(min(x_inter, self.b), self.a)
                y_inter = fi_p - L * abs(x_inter - xi_p)
                accurate_x.append(x_inter)
                accurate_y.append(y_inter)
        accurate_x.append(self.b)
        accurate_y.append(self.func(self.b) - L * abs(self.b - xi[np.argmax(fi - L * abs(self.b - xi))]))

        # Sort and unique
        combined = sorted(zip(accurate_x, accurate_y), key=lambda p: p[0])
        xs = [p[0] for p in combined]
        ys = [p[1] for p in combined]
        # remove nearly duplicate x
        xs_unique, ys_unique = [], []
        last_x = None
        for x, y in zip(xs, ys):
            if last_x is None or abs(x - last_x) > 1e-9:
                xs_unique.append(x)
                ys_unique.append(y)
                last_x = x
        return np.array(xs_unique), np.array(ys_unique)
    
    def visualize(self, x_min: float, f_min: float, save_path: str = 'optimization_results.png', y_tick_step: float = None, y_clip_bottom_percent: float = 0.0, y_clip_top_percent: float = 0.0):
        """научный стиль визуализации"""
        fig = plt.figure(figsize=(15, 5))
        
        x_plot = np.linspace(self.a, self.b, 3000)
        y_plot = np.array([self.func(x) for x in x_plot])
        
        # Создаем сетку графиков: основной график и панель с информацией
        gs = fig.add_gridspec(1, 2, width_ratios=[4, 1])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        
        # Основной график
        ax1.plot(x_plot, y_plot, color='tab:blue', linewidth=2.0, label='f(x)')
        
        if self.history:
            # Показываем плотность точек
            all_x = [p[0] for points in self.history for p in points]
            ax1.hist(all_x, bins=50, density=True, alpha=0.3, color='red', label='плотность поиска')
            
            # Финальные точки
            final_x = [p[0] for p in self.history[-1]]
            final_y = [p[1] for p in self.history[-1]]
            ax1.plot(final_x, final_y, 'ro', markersize=4, alpha=0.9, label='финальные точки')
            # Полигональная нижняя огибающая (точные вершины)
            env_x, env_y = self._envelope_vertices(self.history[-1], n_grid=2000)
            if len(env_x) > 1:
                # Plot envelope as connected polyline (sharp corners), not as steps
                ax1.plot(env_x, env_y, color='tab:red', linewidth=2.4, linestyle='--', marker='o', markersize=5, label='нижняя огибающая (ломаная)')

            # Ломаная через точки выборки (для наглядности поиска)
            pts_sorted = sorted(self.history[-1], key=lambda p: p[0])
            poly_x = [p[0] for p in pts_sorted]
            poly_y = [p[1] for p in pts_sorted]
            # Show sample points connected with a subtle line
            ax1.plot(poly_x, poly_y, color='tab:orange', linestyle='-.', linewidth=0.9, alpha=0.7, label='ломаная выборки')

            # Поддерживающие точки (где огибающая касается функции) - отметим их
            # для каждого пробного xi проверим, есть ли он в опоре: fi - L*|x-xi| == envelope(x)
            env = self._lower_envelope(x_plot, self.history[-1])
            support_x = []
            support_y = []
            for p in pts_sorted:
                xi_p = p[0]
                fi_p = p[1]
                # get envelope value at xi_p
                idx = int(round((xi_p - self.a) / (self.b - self.a) * (len(x_plot) - 1)))
                val = env[idx]
                if abs(val - fi_p) < 1e-6:
                    support_x.append(xi_p)
                    support_y.append(fi_p)
            if support_x:
                ax1.plot(support_x, support_y, 'ks', markersize=6, label='поддерживающие точки')

            # Отметим финальный интервал, если он есть
            if hasattr(self, 'final_interval') and self.final_interval is not None:
                xi, xj, fi, fj = self.final_interval
                ax1.axvspan(xi, xj, color='green', alpha=0.15, label='финальный интервал (eps)')
        
        ax1.plot(x_min, f_min, 'ko', markersize=8, label=f'минимум: x={x_min:.4f}')
        ax1.set_xlabel('x')
        ax1.set_ylabel('f(x)')
        # set y-axis tick step if requested
        if y_tick_step is not None and y_tick_step > 0:
            ax1.yaxis.set_major_locator(MultipleLocator(y_tick_step))

        # y-zoom: clip bottom/top percentiles if requested, to zoom in on the plot
        if (y_clip_bottom_percent and y_clip_bottom_percent > 0) or (y_clip_top_percent and y_clip_top_percent > 0):
            # compute percentiles over the plotted function values
            lo = np.percentile(y_plot, 100.0 * float(y_clip_bottom_percent)) if y_clip_bottom_percent and y_clip_bottom_percent > 0 else np.min(y_plot)
            hi = np.percentile(y_plot, 100.0 * (1.0 - float(y_clip_top_percent))) if y_clip_top_percent and y_clip_top_percent > 0 else np.max(y_plot)
            if hi > lo:
                padding = 0.05 * (hi - lo)
                ax1.set_ylim(lo - padding, hi + padding)
        ax1.legend()
        ax1.grid(True, alpha=0.2)
        
        # Информация
        ax2.axis('off')
        info_text = f"""Algorithm: Piyavsky
    Function: {self.func_str}
    Domain: [{self.a}, {self.b}]
    Precision: ε={self.eps}

    Minimum:
    x* = {x_min:.6f}
    f* = {f_min:.6f}

    Iterations: {self.iterations}
    Time: {self.computation_time:.3f}s"""
        ax2.text(0.1, 0.9, info_text, fontsize=10, family='monospace', verticalalignment='top')
        
        # ПРАВИЛЬНЫЙ график сходимости
        # (Removed convergence subplot)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def print_results(self, x_min: float, f_min: float):
        """вывод результатов в текстовом виде"""
        print("="*60)
        print("результаты глобальной оптимизации")
        print("="*60)
        print(f"функция: f(x) = {self.func_str}")
        print(f"отрезок: [{self.a}, {self.b}]")
        print(f"точность: eps = {self.eps}")
        print("="*60)
        print(f"найденный аргумент минимума: x* = {x_min:.8f}")
        print(f"значение функции в минимуме: f(x*) = {f_min:.8f}")
        print("="*60)
        print(f"число итераций: {self.iterations}")
        print(f"время выполнения: {self.computation_time:.6f} секунд")
        print(f"число пробных точек: {len(self.history[-1]) if self.history else 0}")
        print("="*60)


def main():
    """главная функция для тестирования"""
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


    print("\n" + "="*80)
    print("лабораторная работа: поиск глобального экстремума")
    print("="*80 + "\n")

    # тест 1: функция растригина
    print("\n" + "="*80)
    print("тест 1: функция растригина")
    print("="*80 + "\n")

    func_rastrigin = "10 + x**2 - 10*cos(2*pi*x)"
    optimizer1 = GlobalOptimizer(func_rastrigin, -5.0, 5.0, 0.01)
    x_min1, f_min1 = optimizer1.optimize()
    optimizer1.print_results(x_min1, f_min1)
    # For Rastrigin, clip the bottom 5% of plotted values so the plot looks closer and doesn't show extreme low tails
    optimizer1.visualize(x_min1, f_min1, 'rastrigin_result.png', y_tick_step=5, y_clip_bottom_percent=0.05)

    # тест 2: функция экли - ИСПРАВЛЕННАЯ ВЕРСИЯ
    print("\n" + "="*80)
    print("тест 2: функция экли")
    print("="*80 + "\n")

    # ВАРИАНТ 1: Более резкая версия
    # func_ackley = "-20*exp(-0.2*sqrt(x**2)) - exp(cos(2*pi*x)) + 20 + np.e"
    
    # ВАРИАНТ 2: Еще более выраженная (если нужна большая амплитуда)
    func_ackley = "-30*exp(-0.1*sqrt(x**2)) - 2*exp(cos(3*pi*x)) + 30 + np.e"
    
    optimizer2 = GlobalOptimizer(func_ackley, -5.0, 5.0, 0.01)
    x_min2, f_min2 = optimizer2.optimize()
    optimizer2.print_results(x_min2, f_min2)
    optimizer2.visualize(x_min2, f_min2, 'ackley_result.png', y_tick_step=2, y_clip_bottom_percent=0.02)

    # тест 3: простая функция
    print("\n" + "="*80)
    print("тест 3: функция x + sin(пx)")
    print("="*80 + "\n")

    func_simple = "x + sin(3.14159*x)"
    optimizer3 = GlobalOptimizer(func_simple, -2.0, 6.0, 0.01)
    x_min3, f_min3 = optimizer3.optimize()
    optimizer3.print_results(x_min3, f_min3)
    optimizer3.visualize(x_min3, f_min3, 'simple_result.png', y_tick_step=1)

    # сводная таблица
    print("\n" + "="*80)
    print("сводная таблица результатов")
    print("="*80)
    print(f"{'функция':<30} {'x_min':<15} {'f_min':<15} {'итерации':<12} {'время (с)':<12}")
    print("-"*80)
    print(f"{'растригин':<30} {x_min1:<15.6f} {f_min1:<15.6f} {optimizer1.iterations:<12} {optimizer1.computation_time:<12.4f}")
    print(f"{'экли':<30} {x_min2:<15.6f} {f_min2:<15.6f} {optimizer2.iterations:<12} {optimizer2.computation_time:<12.4f}")
    print(f"{'x + sin(пx)':<30} {x_min3:<15.6f} {f_min3:<15.6f} {optimizer3.iterations:<12} {optimizer3.computation_time:<12.4f}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()