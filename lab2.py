"""
Лабораторна робота №2
Оцінювання впливу зовнішніх та внутрішніх факторів на смарт-підприємство "Coop"
"""

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class AHPAnalyzer:
    """Клас для аналізу ієрархій методом аналізу ієрархій (AHP)"""
    
    def __init__(self):
        # Назви груп впливів (рівень 2)
        self.groups = {
            'E': 'Економічні',
            'R': 'Ринкові',
            'K': 'Конкурентні',
            'B': 'Виробничо-технологічні'
        }
        
        # Фактори впливу для кожної групи (рівень 3)
        self.factors = {
            'E': [
                'E1: Купівельна спроможність населення',
                'E2: Рівень інфляції',
                'E3: Рівень середньої заробітної плати',
                'E4: Податкова політика'
            ],
            'R': [
                'R1: Місткість ринку роздрібної торгівлі',
                'R2: Лояльність покупців',
                'R3: Географічне розташування магазинів',
                'R4: Асортиментна політика'
            ],
            'K': [
                'K1: Кількість конкурентів (АТБ, Сільпо, Вопак)',
                'K2: Цінова конкуренція',
                'K3: Програми лояльності конкурентів',
                'K4: Мережа магазинів конкурентів'
            ],
            'B': [
                'B1: Рівень автоматизації процесів',
                'B2: Логістична інфраструктура',
                'B3: Системи управління запасами',
                'B4: Касові системи та POS-термінали'
            ]
        }
        
    def create_pairwise_matrix(self, weights):
        """
        Створення матриці попарних порівнянь на основі ваг важливості
        weights: список ваг важливості факторів
        """
        n = len(weights)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i][j] = 1
                elif i < j:
                    # Порівняння i-го фактора з j-им
                    matrix[i][j] = weights[i] / weights[j]
                else:
                    # Властивість оберненої симетричності
                    matrix[i][j] = 1 / matrix[j][i]
        
        return matrix
    
    def calculate_eigenvector(self, matrix):
        """
        Обчислення власного вектора матриці
        Метод геометричного середнього
        """
        n = matrix.shape[0]
        eigenvector = np.zeros(n)
        
        for i in range(n):
            # Добуток елементів рядка
            product = np.prod(matrix[i, :])
            # Корінь n-ї степені
            eigenvector[i] = product ** (1/n)
        
        return eigenvector
    
    def normalize_vector(self, vector):
        """Нормалізація вектора"""
        return vector / np.sum(vector)
    
    def calculate_priority_vector(self, matrix):
        """
        Обчислення вектора пріоритетів
        1. Обчислення власного вектора
        2. Нормалізація
        """
        eigenvector = self.calculate_eigenvector(matrix)
        priority_vector = self.normalize_vector(eigenvector)
        return priority_vector
    
    def calculate_lambda_max(self, matrix, priority_vector):
        """Обчислення максимального власного значення λmax"""
        n = matrix.shape[0]
        column_sums = np.sum(matrix, axis=0)
        lambda_max = np.dot(column_sums, priority_vector)
        return lambda_max
    
    def calculate_consistency_index(self, matrix, priority_vector):
        """
        Обчислення індексу узгодженості (ІУ)
        ІУ = (λmax - n) / (n - 1)
        """
        n = matrix.shape[0]
        lambda_max = self.calculate_lambda_max(matrix, priority_vector)
        
        if n == 1:
            return 0
        
        ci = (lambda_max - n) / (n - 1)
        return ci, lambda_max
    
    def calculate_consistency_ratio(self, ci, n):
        """
        Обчислення відношення узгодженості (ВУ)
        ВУ = ІУ / ВВ, де ВВ - випадкова узгодженість
        """
        # Таблиця випадкової узгодженості
        random_index = {
            1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12,
            6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49
        }
        
        ri = random_index.get(n, 1.49)
        if ri == 0:
            return 0
        
        cr = ci / ri
        return cr
    
    def calculate_integral_indicator(self, group_priorities, factor_priorities_dict):
        """
        Обчислення узагальненого інтегрального показника впливу Iв
        Iв = An × G
        де An - вектор локальних пріоритетів груп (рівень 2)
        G - вектор глобальних пріоритетів (максимальні значення з рівня 3)
        """
        groups_list = list(self.groups.keys())
        global_priorities = []
        
        for group in groups_list:
            # Знаходимо максимальний пріоритет у кожній групі
            max_priority = np.max(factor_priorities_dict[group])
            global_priorities.append(max_priority)
        
        global_vector = np.array(global_priorities)
        integral_indicator = np.dot(group_priorities, global_vector)
        
        return integral_indicator, global_vector


class AHPAnalysisGUI:
    """Графічний інтерфейс для аналізу впливу факторів"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Аналіз впливу факторів на смарт-підприємство 'Coop'")
        self.root.geometry("1200x800")
        
        self.analyzer = AHPAnalyzer()
        
        # Збереження результатів
        self.group_priorities = None
        self.factor_priorities = {}
        self.group_matrix = None
        self.factor_matrices = {}
        
        self.create_widgets()
        
    def create_widgets(self):
        """Створення віджетів інтерфейсу"""
        
        # Notebook для вкладок
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Вкладка 1: Введення даних для груп впливів
        self.tab_groups = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_groups, text='Групи впливів (Рівень 2)')
        self.create_groups_tab()
        
        # Вкладка 2: Введення даних для факторів впливу
        self.tab_factors = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_factors, text='Фактори впливу (Рівень 3)')
        self.create_factors_tab()
        
        # Вкладка 3: Результати
        self.tab_results = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_results, text='Результати аналізу')
        self.create_results_tab()
        
        # Вкладка 4: Візуалізація
        self.tab_visualization = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_visualization, text='Візуалізація')
        self.create_visualization_tab()
        
    def create_groups_tab(self):
        """Вкладка для введення ваг груп впливів"""
        
        frame = ttk.LabelFrame(self.tab_groups, text="Введіть ваги важливості груп впливів", padding=10)
        frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Інструкція
        instruction = ttk.Label(frame, text="""
Шкала ваг важливості:
1 - Рівна важливість
3 - Помірна перевага
5 - Суттєва перевага
7 - Значна перевага
9 - Дуже велика перевага
2, 4, 6, 8 - Проміжні значення
        """, justify='left', font=('Arial', 10))
        instruction.pack(anchor='w', pady=(0, 10))
        
        # Створення полів для введення
        self.group_entries = {}
        
        input_frame = ttk.Frame(frame)
        input_frame.pack(fill='x', pady=10)
        
        groups_list = list(self.analyzer.groups.keys())
        for i, (code, name) in enumerate(self.analyzer.groups.items()):
            row_frame = ttk.Frame(input_frame)
            row_frame.pack(fill='x', pady=5)
            
            label = ttk.Label(row_frame, text=f"{code} - {name}:", width=40)
            label.pack(side='left', padx=5)
            
            entry = ttk.Entry(row_frame, width=10)
            entry.pack(side='left', padx=5)
            entry.insert(0, str(5 - i))  # Значення за замовчуванням
            
            self.group_entries[code] = entry
        
        # Кнопка обчислення
        btn_calculate = ttk.Button(frame, text="Обчислити вектор пріоритетів для груп",
                                   command=self.calculate_group_priorities)
        btn_calculate.pack(pady=10)
        
        # Область для результатів
        self.group_results_text = scrolledtext.ScrolledText(frame, height=15, width=80)
        self.group_results_text.pack(pady=10, fill='both', expand=True)
        
    def create_factors_tab(self):
        """Вкладка для введення ваг факторів впливу"""
        
        # Використовуємо Notebook для підвкладок для кожної групи
        self.factors_notebook = ttk.Notebook(self.tab_factors)
        self.factors_notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.factor_entries = {}
        self.factor_results_texts = {}
        
        for group_code, group_name in self.analyzer.groups.items():
            # Створюємо вкладку для кожної групи
            tab = ttk.Frame(self.factors_notebook)
            self.factors_notebook.add(tab, text=f"{group_code}: {group_name}")
            
            frame = ttk.LabelFrame(tab, text=f"Фактори групи {group_code}", padding=10)
            frame.pack(fill='both', expand=True, padx=10, pady=10)
            
            # Інструкція
            instruction = ttk.Label(frame, text=f"Введіть ваги важливості для факторів групи {group_code}",
                                  font=('Arial', 10, 'bold'))
            instruction.pack(anchor='w', pady=(0, 10))
            
            # Поля для введення
            self.factor_entries[group_code] = {}
            
            input_frame = ttk.Frame(frame)
            input_frame.pack(fill='x', pady=10)
            
            for i, factor_name in enumerate(self.analyzer.factors[group_code]):
                row_frame = ttk.Frame(input_frame)
                row_frame.pack(fill='x', pady=3)
                
                label = ttk.Label(row_frame, text=factor_name, width=50)
                label.pack(side='left', padx=5)
                
                entry = ttk.Entry(row_frame, width=10)
                entry.pack(side='left', padx=5)
                entry.insert(0, str(5 - i))  # Значення за замовчуванням
                
                self.factor_entries[group_code][i] = entry
            
            # Кнопка обчислення
            btn_calculate = ttk.Button(frame, text=f"Обчислити вектор пріоритетів для групи {group_code}",
                                      command=lambda gc=group_code: self.calculate_factor_priorities(gc))
            btn_calculate.pack(pady=10)
            
            # Область для результатів
            results_text = scrolledtext.ScrolledText(frame, height=12, width=80)
            results_text.pack(pady=10, fill='both', expand=True)
            self.factor_results_texts[group_code] = results_text
        
        # Кнопка обчислення для всіх груп
        btn_frame = ttk.Frame(self.tab_factors)
        btn_frame.pack(side='bottom', pady=10)
        
        btn_calculate_all = ttk.Button(btn_frame, text="Обчислити всі вектори пріоритетів",
                                      command=self.calculate_all_factor_priorities,
                                      style='Accent.TButton')
        btn_calculate_all.pack()
        
    def create_results_tab(self):
        """Вкладка для відображення результатів"""
        
        frame = ttk.LabelFrame(self.tab_results, text="Результати аналізу", padding=10)
        frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Кнопка обчислення інтегрального показника
        btn_calculate = ttk.Button(frame, text="Обчислити узагальнений інтегральний показник впливу",
                                   command=self.calculate_integral_indicator,
                                   style='Accent.TButton')
        btn_calculate.pack(pady=10)
        
        # Область для результатів
        self.results_text = scrolledtext.ScrolledText(frame, height=30, width=100, font=('Courier', 10))
        self.results_text.pack(pady=10, fill='both', expand=True)
        
    def create_visualization_tab(self):
        """Вкладка для візуалізації результатів"""
        
        frame = ttk.Frame(self.tab_visualization)
        frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Кнопка візуалізації
        btn_visualize = ttk.Button(frame, text="Побудувати графіки",
                                   command=self.visualize_results)
        btn_visualize.pack(pady=10)
        
        # Canvas для графіків
        self.visualization_frame = ttk.Frame(frame)
        self.visualization_frame.pack(fill='both', expand=True)
        
    def calculate_group_priorities(self):
        """Обчислення векторів пріоритетів для груп впливів"""
        try:
            # Отримання ваг від користувача
            weights = []
            groups_list = list(self.analyzer.groups.keys())
            
            for code in groups_list:
                weight = float(self.group_entries[code].get())
                weights.append(weight)
            
            weights = np.array(weights)
            
            # Створення матриці попарних порівнянь
            self.group_matrix = self.analyzer.create_pairwise_matrix(weights)
            
            # Обчислення вектора пріоритетів
            self.group_priorities = self.analyzer.calculate_priority_vector(self.group_matrix)
            
            # Обчислення індексу узгодженості
            ci, lambda_max = self.analyzer.calculate_consistency_index(self.group_matrix, self.group_priorities)
            cr = self.analyzer.calculate_consistency_ratio(ci, len(weights))
            
            # Виведення результатів
            self.group_results_text.delete(1.0, tk.END)
            
            result = "=" * 70 + "\n"
            result += "РЕЗУЛЬТАТИ АНАЛІЗУ ГРУП ВПЛИВІВ (РІВЕНЬ 2)\n"
            result += "=" * 70 + "\n\n"
            
            result += "1. ВВЕДЕНІ ВАГИ ВАЖЛИВОСТІ:\n"
            result += "-" * 70 + "\n"
            for i, code in enumerate(groups_list):
                result += f"   {code} ({self.analyzer.groups[code]}): {weights[i]:.2f}\n"
            
            result += "\n2. МАТРИЦЯ ПОПАРНИХ ПОРІВНЯНЬ:\n"
            result += "-" * 70 + "\n"
            result += "     " + "    ".join([f"{c:>6}" for c in groups_list]) + "\n"
            for i, code in enumerate(groups_list):
                row = f"{code:>3}  " + "  ".join([f"{self.group_matrix[i][j]:>6.3f}" for j in range(len(groups_list))])
                result += row + "\n"
            
            result += "\n3. ВЕКТОР ПРІОРИТЕТІВ:\n"
            result += "-" * 70 + "\n"
            for i, code in enumerate(groups_list):
                result += f"   {code} ({self.analyzer.groups[code]}): {self.group_priorities[i]:.4f} ({self.group_priorities[i]*100:.2f}%)\n"
            
            result += f"\n4. ПОКАЗНИКИ УЗГОДЖЕНОСТІ:\n"
            result += "-" * 70 + "\n"
            result += f"   λmax = {lambda_max:.4f}\n"
            result += f"   Індекс узгодженості (ІУ) = {ci:.4f}\n"
            result += f"   Відношення узгодженості (ВУ) = {cr:.4f}\n"
            
            if cr < 0.10:
                result += f"   ✓ Матриця є узгодженою (ВУ < 0.10)\n"
            else:
                result += f"   ✗ УВАГА: Матриця недостатньо узгоджена (ВУ ≥ 0.10)\n"
            
            result += "\n" + "=" * 70 + "\n"
            
            self.group_results_text.insert(1.0, result)
            
            messagebox.showinfo("Успіх", "Вектор пріоритетів для груп обчислено успішно!")
            
        except ValueError:
            messagebox.showerror("Помилка", "Будь ласка, введіть коректні числові значення!")
        except Exception as e:
            messagebox.showerror("Помилка", f"Помилка при обчисленні: {str(e)}")
    
    def calculate_factor_priorities(self, group_code):
        """Обчислення векторів пріоритетів для факторів певної групи"""
        try:
            # Отримання ваг від користувача
            weights = []
            for i in range(len(self.analyzer.factors[group_code])):
                weight = float(self.factor_entries[group_code][i].get())
                weights.append(weight)
            
            weights = np.array(weights)
            
            # Створення матриці попарних порівнянь
            matrix = self.analyzer.create_pairwise_matrix(weights)
            self.factor_matrices[group_code] = matrix
            
            # Обчислення вектора пріоритетів
            priorities = self.analyzer.calculate_priority_vector(matrix)
            self.factor_priorities[group_code] = priorities
            
            # Обчислення індексу узгодженості
            ci, lambda_max = self.analyzer.calculate_consistency_index(matrix, priorities)
            cr = self.analyzer.calculate_consistency_ratio(ci, len(weights))
            
            # Виведення результатів
            results_text = self.factor_results_texts[group_code]
            results_text.delete(1.0, tk.END)
            
            result = "=" * 70 + "\n"
            result += f"РЕЗУЛЬТАТИ АНАЛІЗУ ГРУПИ {group_code} ({self.analyzer.groups[group_code]})\n"
            result += "=" * 70 + "\n\n"
            
            result += "1. ВВЕДЕНІ ВАГИ ВАЖЛИВОСТІ:\n"
            result += "-" * 70 + "\n"
            for i, factor_name in enumerate(self.analyzer.factors[group_code]):
                result += f"   {factor_name}: {weights[i]:.2f}\n"
            
            result += "\n2. ВЕКТОР ПРІОРИТЕТІВ:\n"
            result += "-" * 70 + "\n"
            for i, factor_name in enumerate(self.analyzer.factors[group_code]):
                result += f"   {factor_name}: {priorities[i]:.4f} ({priorities[i]*100:.2f}%)\n"
            
            result += f"\n3. ПОКАЗНИКИ УЗГОДЖЕНОСТІ:\n"
            result += "-" * 70 + "\n"
            result += f"   λmax = {lambda_max:.4f}\n"
            result += f"   Індекс узгодженості (ІУ) = {ci:.4f}\n"
            result += f"   Відношення узгодженості (ВУ) = {cr:.4f}\n"
            
            if cr < 0.10:
                result += f"   ✓ Матриця є узгодженою (ВУ < 0.10)\n"
            else:
                result += f"   ✗ УВАГА: Матриця недостатньо узгоджена (ВУ ≥ 0.10)\n"
            
            result += "\n" + "=" * 70 + "\n"
            
            results_text.insert(1.0, result)
            
            messagebox.showinfo("Успіх", f"Вектор пріоритетів для групи {group_code} обчислено успішно!")
            
        except ValueError:
            messagebox.showerror("Помилка", "Будь ласка, введіть коректні числові значення!")
        except Exception as e:
            messagebox.showerror("Помилка", f"Помилка при обчисленні: {str(e)}")
    
    def calculate_all_factor_priorities(self):
        """Обчислення векторів пріоритетів для всіх груп факторів"""
        groups_list = list(self.analyzer.groups.keys())
        
        for group_code in groups_list:
            self.calculate_factor_priorities(group_code)
        
        messagebox.showinfo("Успіх", "Вектори пріоритетів для всіх груп обчислено успішно!")
    
    def calculate_integral_indicator(self):
        """Обчислення узагальненого інтегрального показника впливу"""
        try:
            # Перевірка, чи обчислені всі необхідні вектори
            if self.group_priorities is None:
                messagebox.showerror("Помилка", "Спочатку обчисліть вектор пріоритетів для груп впливів!")
                return
            
            groups_list = list(self.analyzer.groups.keys())
            
            for group_code in groups_list:
                if group_code not in self.factor_priorities:
                    messagebox.showerror("Помилка", 
                                       f"Спочатку обчисліть вектор пріоритетів для групи {group_code}!")
                    return
            
            # Обчислення інтегрального показника
            integral_indicator, global_vector = self.analyzer.calculate_integral_indicator(
                self.group_priorities, self.factor_priorities
            )
            
            # Формування звіту
            self.results_text.delete(1.0, tk.END)
            
            report = "╔" + "═" * 78 + "╗\n"
            report += "║" + " " * 20 + "ЗВІТ ПРО АНАЛІЗ ВПЛИВУ ФАКТОРІВ" + " " * 27 + "║\n"
            report += "║" + " " * 18 + "НА СМАРТ-ПІДПРИЄМСТВО 'Coop'" + " " * 31 + "║\n"
            report += "╚" + "═" * 78 + "╝\n\n"
            
            report += "┌" + "─" * 78 + "┐\n"
            report += "│ 1. ПРІОРИТЕТИ ГРУП ВПЛИВІВ (РІВЕНЬ 2)" + " " * 40 + "│\n"
            report += "└" + "─" * 78 + "┘\n\n"
            
            for i, code in enumerate(groups_list):
                bar_length = int(self.group_priorities[i] * 50)
                bar = "█" * bar_length + "░" * (50 - bar_length)
                report += f"   {code} {self.analyzer.groups[code]:30} {self.group_priorities[i]:.4f} {bar}\n"
            
            report += "\n┌" + "─" * 78 + "┐\n"
            report += "│ 2. ПРІОРИТЕТИ ФАКТОРІВ ВПЛИВУ (РІВЕНЬ 3)" + " " * 37 + "│\n"
            report += "└" + "─" * 78 + "┘\n\n"
            
            for group_code in groups_list:
                report += f"\n   ┌─ {group_code}: {self.analyzer.groups[group_code]}\n"
                priorities = self.factor_priorities[group_code]
                
                for i, factor_name in enumerate(self.analyzer.factors[group_code]):
                    bar_length = int(priorities[i] * 40)
                    bar = "█" * bar_length + "░" * (40 - bar_length)
                    report += f"   │  {factor_name[:45]:45} {priorities[i]:.4f} {bar}\n"
                
                max_idx = np.argmax(priorities)
                report += f"   └─ Домінуючий фактор: {self.analyzer.factors[group_code][max_idx]}\n"
            
            report += "\n┌" + "─" * 78 + "┐\n"
            report += "│ 3. ВЕКТОР ГЛОБАЛЬНИХ ПРІОРИТЕТІВ" + " " * 44 + "│\n"
            report += "└" + "─" * 78 + "┘\n\n"
            
            report += "   (Максимальні пріоритети з кожної групи факторів)\n\n"
            
            for i, code in enumerate(groups_list):
                max_idx = np.argmax(self.factor_priorities[code])
                max_factor = self.analyzer.factors[code][max_idx]
                report += f"   {code}: {global_vector[i]:.4f} - {max_factor}\n"
            
            report += "\n┌" + "─" * 78 + "┐\n"
            report += "│ 4. УЗАГАЛЬНЕНИЙ ІНТЕГРАЛЬНИЙ ПОКАЗНИК ВПЛИВУ" + " " * 32 + "│\n"
            report += "└" + "─" * 78 + "┘\n\n"
            
            report += f"   Iв = An × G = {integral_indicator:.6f}\n\n"
            
            report += "   Розрахунок:\n"
            for i, code in enumerate(groups_list):
                report += f"   + {self.group_priorities[i]:.4f} × {global_vector[i]:.4f} "
                report += f"({code}: {self.analyzer.groups[code]})\n"
            
            report += f"\n   Сума = {integral_indicator:.6f}\n"
            
            report += "\n┌" + "─" * 78 + "┐\n"
            report += "│ 5. ВИСНОВКИ ТА РЕКОМЕНДАЦІЇ" + " " * 49 + "│\n"
            report += "└" + "─" * 78 + "┘\n\n"
            
            # Визначення найбільш впливової групи
            max_group_idx = np.argmax(self.group_priorities)
            max_group_code = groups_list[max_group_idx]
            
            report += f"   ■ Найвпливовіша група факторів:\n"
            report += f"     {max_group_code}: {self.analyzer.groups[max_group_code]} "
            report += f"({self.group_priorities[max_group_idx]:.2%})\n\n"
            
            # Топ-3 найбільш впливових факторів загалом
            all_factors_weighted = []
            for i, code in enumerate(groups_list):
                for j, priority in enumerate(self.factor_priorities[code]):
                    weighted_priority = self.group_priorities[i] * priority
                    all_factors_weighted.append({
                        'group': code,
                        'factor': self.analyzer.factors[code][j],
                        'priority': weighted_priority
                    })
            
            all_factors_weighted.sort(key=lambda x: x['priority'], reverse=True)
            
            report += "   ■ Топ-3 найбільш впливових факторів:\n"
            for i in range(min(3, len(all_factors_weighted))):
                factor_info = all_factors_weighted[i]
                report += f"     {i+1}. [{factor_info['group']}] {factor_info['factor']} "
                report += f"({factor_info['priority']:.4f})\n"
            
            report += f"\n   ■ Інтегральний показник впливу Iв = {integral_indicator:.6f}\n"
            
            if integral_indicator > 0.5:
                report += "     Високий рівень впливу факторів на підприємство.\n"
            elif integral_indicator > 0.3:
                report += "     Помірний рівень впливу факторів на підприємство.\n"
            else:
                report += "     Низький рівень впливу факторів на підприємство.\n"
            
            report += "\n" + "═" * 80 + "\n"
            report += "Дата аналізу: " + "2025-11-01" + "\n"
            report += "Підприємство: Coop - кооперативна роздрібна мережа\n"
            report += "═" * 80 + "\n"
            
            self.results_text.insert(1.0, report)
            
            messagebox.showinfo("Успіх", 
                              f"Узагальнений інтегральний показник впливу обчислено!\n\nIв = {integral_indicator:.6f}")
            
        except Exception as e:
            messagebox.showerror("Помилка", f"Помилка при обчисленні: {str(e)}")
    
    def visualize_results(self):
        """Візуалізація результатів аналізу"""
        try:
            # Очищення попередніх графіків
            for widget in self.visualization_frame.winfo_children():
                widget.destroy()
            
            if self.group_priorities is None or not self.factor_priorities:
                messagebox.showerror("Помилка", "Спочатку виконайте обчислення!")
                return
            
            # Створення фігури з підграфіками
            fig = Figure(figsize=(14, 10))
            
            # 1. Пріоритети груп впливів (кругова діаграма)
            ax1 = fig.add_subplot(2, 3, 1)
            groups_list = list(self.analyzer.groups.keys())
            labels = [f"{code}\n{self.analyzer.groups[code]}" for code in groups_list]
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
            
            wedges, texts, autotexts = ax1.pie(self.group_priorities, labels=labels, autopct='%1.1f%%',
                                                colors=colors, startangle=90)
            ax1.set_title('Пріоритети груп впливів', fontsize=12, fontweight='bold')
            
            # 2. Пріоритети груп впливів (стовпчикова діаграма)
            ax2 = fig.add_subplot(2, 3, 2)
            x_pos = np.arange(len(groups_list))
            bars = ax2.bar(x_pos, self.group_priorities, color=colors, alpha=0.7, edgecolor='black')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels([f"{code}" for code in groups_list])
            ax2.set_ylabel('Пріоритет')
            ax2.set_title('Порівняння пріоритетів груп', fontsize=12, fontweight='bold')
            ax2.grid(axis='y', alpha=0.3)
            
            # Додавання значень на стовпчики
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=9)
            
            # 3-6. Пріоритети факторів для кожної групи
            for idx, group_code in enumerate(groups_list):
                ax = fig.add_subplot(2, 3, idx + 3)
                
                priorities = self.factor_priorities[group_code]
                factor_labels = [f.split(':')[0] for f in self.analyzer.factors[group_code]]
                
                y_pos = np.arange(len(factor_labels))
                bars = ax.barh(y_pos, priorities, color=colors[idx], alpha=0.7, edgecolor='black')
                
                ax.set_yticks(y_pos)
                ax.set_yticklabels(factor_labels, fontsize=8)
                ax.set_xlabel('Пріоритет', fontsize=9)
                ax.set_title(f'{group_code}: {self.analyzer.groups[group_code]}', 
                            fontsize=10, fontweight='bold')
                ax.grid(axis='x', alpha=0.3)
                
                # Додавання значень
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    ax.text(width, bar.get_y() + bar.get_height()/2.,
                           f'{width:.3f}',
                           ha='left', va='center', fontsize=8)
            
            fig.tight_layout(pad=2.0)
            
            # Додавання canvas до інтерфейсу
            canvas = FigureCanvasTkAgg(fig, master=self.visualization_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            
            messagebox.showinfo("Успіх", "Візуалізація побудована успішно!")
            
        except Exception as e:
            messagebox.showerror("Помилка", f"Помилка при візуалізації: {str(e)}")


def main():
    """Головна функція програми"""
    root = tk.Tk()
    
    # Налаштування стилю
    style = ttk.Style()
    style.theme_use('clam')
    
    app = AHPAnalysisGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
