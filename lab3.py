"""
Лабораторна робота №2 (Розширена версія)
Оцінювання впливу зовнішніх та внутрішніх факторів на смарт-підприємство "Coop"
з квартальним аналізом, прогнозуванням та визначенням трендів
"""

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


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
            return 0, n
        
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


class QuarterlyAnalyzer:
    """Клас для квартального аналізу та прогнозування"""
    
    def __init__(self):
        self.historical_data = []  # Історичні дані (4 квартали)
        self.predicted_data = []   # Прогнозовані дані (4 квартали)
        self.threshold = 0.0       # Поріг відхилення
        
    def add_quarterly_data(self, quarter_values):
        """Додавання квартальних даних"""
        self.historical_data = quarter_values
        
    def forecast_linear_regression(self, n_future=4):
        """
        Прогнозування методом лінійної регресії
        n_future: кількість періодів для прогнозування
        """
        if len(self.historical_data) < 2:
            raise ValueError("Недостатньо даних для прогнозування")
        
        # Підготовка даних
        X = np.array(range(1, len(self.historical_data) + 1)).reshape(-1, 1)
        y = np.array(self.historical_data)
        
        # Навчання моделі
        model = LinearRegression()
        model.fit(X, y)
        
        # Прогнозування
        future_X = np.array(range(len(self.historical_data) + 1, 
                                  len(self.historical_data) + n_future + 1)).reshape(-1, 1)
        predictions = model.predict(future_X)
        
        self.predicted_data = predictions.tolist()
        return self.predicted_data
    
    def forecast_polynomial_regression(self, n_future=4, degree=2):
        """
        Прогнозування методом поліноміальної регресії
        n_future: кількість періодів для прогнозування
        degree: ступінь полінома
        """
        if len(self.historical_data) < degree + 1:
            raise ValueError("Недостатньо даних для поліноміальної регресії")
        
        # Підготовка даних
        X = np.array(range(1, len(self.historical_data) + 1)).reshape(-1, 1)
        y = np.array(self.historical_data)
        
        # Поліноміальні ознаки
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)
        
        # Навчання моделі
        model = LinearRegression()
        model.fit(X_poly, y)
        
        # Прогнозування
        future_X = np.array(range(len(self.historical_data) + 1, 
                                  len(self.historical_data) + n_future + 1)).reshape(-1, 1)
        future_X_poly = poly.transform(future_X)
        predictions = model.predict(future_X_poly)
        
        self.predicted_data = predictions.tolist()
        return self.predicted_data
    
    def forecast_exponential_smoothing(self, n_future=4, alpha=0.3):
        """
        Прогнозування методом експоненціального згладжування
        n_future: кількість періодів для прогнозування
        alpha: коефіцієнт згладжування (0 < alpha < 1)
        """
        if len(self.historical_data) < 1:
            raise ValueError("Недостатньо даних для прогнозування")
        
        # Ініціалізація
        smoothed = [self.historical_data[0]]
        
        # Згладжування історичних даних
        for i in range(1, len(self.historical_data)):
            value = alpha * self.historical_data[i] + (1 - alpha) * smoothed[i-1]
            smoothed.append(value)
        
        # Прогнозування
        predictions = []
        last_smoothed = smoothed[-1]
        
        for _ in range(n_future):
            predictions.append(last_smoothed)
        
        self.predicted_data = predictions
        return self.predicted_data
    
    def forecast_moving_average(self, n_future=4, window=3):
        """
        Прогнозування методом ковзного середнього
        n_future: кількість періодів для прогнозування
        window: розмір вікна для ковзного середнього
        """
        if len(self.historical_data) < window:
            window = len(self.historical_data)
        
        # Обчислення ковзного середнього для останніх даних
        last_values = self.historical_data[-window:]
        avg = np.mean(last_values)
        
        # Прогнозування (просте повторення середнього)
        predictions = [avg] * n_future
        
        self.predicted_data = predictions
        return self.predicted_data
    
    def calculate_threshold(self, method='std', factor=1.5):
        """
        Обчислення порогового значення
        method: 'std' (стандартне відхилення) або 'range' (діапазон)
        factor: множник для порогу
        """
        if len(self.historical_data) < 2:
            return 0
        
        if method == 'std':
            # Поріг на основі стандартного відхилення
            std_dev = np.std(self.historical_data)
            self.threshold = factor * std_dev
        elif method == 'range':
            # Поріг на основі діапазону
            data_range = np.max(self.historical_data) - np.min(self.historical_data)
            self.threshold = factor * data_range / 2
        else:
            # Відсоток від середнього
            mean_val = np.mean(self.historical_data)
            self.threshold = factor * mean_val
        
        return self.threshold
    
    def compare_forecast_with_real(self, real_values):
        """
        Порівняння прогнозованих значень з реальними
        real_values: реальні значення для порівняння
        """
        if len(real_values) != len(self.predicted_data):
            raise ValueError("Кількість реальних значень не співпадає з прогнозованими")
        
        deviations = []
        trends = []
        
        for i in range(len(real_values)):
            deviation = abs(real_values[i] - self.predicted_data[i])
            deviations.append(deviation)
            
            # Визначення тренду
            if deviation <= self.threshold:
                trend = "Стабільний"
            elif real_values[i] > self.predicted_data[i]:
                trend = "Зростання"
            else:
                trend = "Спадання"
            
            trends.append(trend)
        
        return deviations, trends


class AHPAnalysisGUI:
    """Графічний інтерфейс для аналізу впливу факторів"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Аналіз впливу факторів на смарт-підприємство 'Coop' (Розширена версія)")
        self.root.geometry("1400x900")
        
        self.analyzer = AHPAnalyzer()
        self.quarterly_analyzer = QuarterlyAnalyzer()
        
        # Збереження результатів
        self.group_priorities = None
        self.factor_priorities = {}
        self.group_matrix = None
        self.factor_matrices = {}
        
        # Квартальні дані
        self.quarterly_indicators = []
        self.real_future_values = []
        
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
        
        # Вкладка 5: Квартальний аналіз (НОВА)
        self.tab_quarterly = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_quarterly, text='Квартальний аналіз')
        self.create_quarterly_tab()
        
        # Вкладка 6: Прогнозування (НОВА)
        self.tab_forecast = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_forecast, text='Прогнозування')
        self.create_forecast_tab()
        
        # Вкладка 7: Порівняння трендів (НОВА)
        self.tab_trends = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_trends, text='Порівняння трендів')
        self.create_trends_tab()
        
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
                                      command=self.calculate_all_factor_priorities)
        btn_calculate_all.pack()
        
    def create_results_tab(self):
        """Вкладка для відображення результатів"""
        
        frame = ttk.LabelFrame(self.tab_results, text="Результати аналізу", padding=10)
        frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Кнопка обчислення інтегрального показника
        btn_calculate = ttk.Button(frame, text="Обчислити узагальнений інтегральний показник впливу",
                                   command=self.calculate_integral_indicator)
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
        
    def create_quarterly_tab(self):
        """Вкладка для квартального аналізу"""
        
        frame = ttk.LabelFrame(self.tab_quarterly, text="Квартальний аналіз інтегрального показника", padding=10)
        frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Інструкція
        instruction = ttk.Label(frame, text="""
Введіть значення узагальненого інтегрального показника впливу для 4 кварталів.
Ці значення можуть бути розраховані окремо для кожного кварталу або змінені вручну.
        """, justify='left', font=('Arial', 10))
        instruction.pack(anchor='w', pady=(0, 10))
        
        # Поля для введення квартальних даних
        input_frame = ttk.LabelFrame(frame, text="Історичні дані (4 квартали)", padding=10)
        input_frame.pack(fill='x', pady=10)
        
        self.quarter_entries = {}
        
        for i in range(1, 5):
            row_frame = ttk.Frame(input_frame)
            row_frame.pack(fill='x', pady=5)
            
            label = ttk.Label(row_frame, text=f"Квартал {i} (Q{i}):", width=20)
            label.pack(side='left', padx=5)
            
            entry = ttk.Entry(row_frame, width=15)
            entry.pack(side='left', padx=5)
            # Значення за замовчуванням (приблизні)
            entry.insert(0, str(round(0.30 + i * 0.05, 4)))
            
            self.quarter_entries[i] = entry
            
            # Кнопка автоматичного обчислення для кварталу
            btn_auto = ttk.Button(row_frame, text=f"Обчислити для Q{i}",
                                 command=lambda q=i: self.calculate_for_quarter(q))
            btn_auto.pack(side='left', padx=5)
        
        # Кнопки дій
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill='x', pady=10)
        
        btn_save = ttk.Button(btn_frame, text="Зберегти квартальні дані",
                             command=self.save_quarterly_data)
        btn_save.pack(side='left', padx=5)
        
        btn_visualize = ttk.Button(btn_frame, text="Побудувати графік",
                                  command=self.visualize_quarterly_data)
        btn_visualize.pack(side='left', padx=5)
        
        # Область для результатів
        self.quarterly_results_text = scrolledtext.ScrolledText(frame, height=15, width=100)
        self.quarterly_results_text.pack(pady=10, fill='both', expand=True)
        
    def create_forecast_tab(self):
        """Вкладка для прогнозування"""
        
        frame = ttk.LabelFrame(self.tab_forecast, text="Прогнозування інтегрального показника", padding=10)
        frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Вибір методу прогнозування
        method_frame = ttk.LabelFrame(frame, text="Метод прогнозування", padding=10)
        method_frame.pack(fill='x', pady=10)
        
        self.forecast_method = tk.StringVar(value="linear")
        
        methods = [
            ("Лінійна регресія", "linear"),
            ("Поліноміальна регресія", "polynomial"),
            ("Експоненціальне згладжування", "exponential"),
            ("Ковзне середнє", "moving_average")
        ]
        
        for text, value in methods:
            rb = ttk.Radiobutton(method_frame, text=text, variable=self.forecast_method, value=value)
            rb.pack(anchor='w', pady=2)
        
        # Параметри методу
        params_frame = ttk.LabelFrame(frame, text="Параметри прогнозування", padding=10)
        params_frame.pack(fill='x', pady=10)
        
        # Ступінь полінома
        poly_frame = ttk.Frame(params_frame)
        poly_frame.pack(fill='x', pady=5)
        ttk.Label(poly_frame, text="Ступінь полінома (для поліноміальної регресії):").pack(side='left', padx=5)
        self.poly_degree = ttk.Spinbox(poly_frame, from_=2, to=4, width=10)
        self.poly_degree.set(2)
        self.poly_degree.pack(side='left', padx=5)
        
        # Alpha для експоненціального згладжування
        alpha_frame = ttk.Frame(params_frame)
        alpha_frame.pack(fill='x', pady=5)
        ttk.Label(alpha_frame, text="Коефіцієнт згладжування α (0-1):").pack(side='left', padx=5)
        self.alpha_entry = ttk.Entry(alpha_frame, width=10)
        self.alpha_entry.insert(0, "0.3")
        self.alpha_entry.pack(side='left', padx=5)
        
        # Вікно для ковзного середнього
        window_frame = ttk.Frame(params_frame)
        window_frame.pack(fill='x', pady=5)
        ttk.Label(window_frame, text="Розмір вікна (для ковзного середнього):").pack(side='left', padx=5)
        self.window_size = ttk.Spinbox(window_frame, from_=2, to=4, width=10)
        self.window_size.set(3)
        self.window_size.pack(side='left', padx=5)
        
        # Кнопка прогнозування
        btn_forecast = ttk.Button(frame, text="Виконати прогнозування на 4 квартали",
                                 command=self.perform_forecast)
        btn_forecast.pack(pady=10)
        
        # Візуалізація
        btn_visualize = ttk.Button(frame, text="Побудувати графік прогнозу",
                                  command=self.visualize_forecast)
        btn_visualize.pack(pady=5)
        
        # Область для результатів
        self.forecast_results_text = scrolledtext.ScrolledText(frame, height=15, width=100)
        self.forecast_results_text.pack(pady=10, fill='both', expand=True)
        
    def create_trends_tab(self):
        """Вкладка для порівняння трендів"""
        
        frame = ttk.LabelFrame(self.tab_trends, text="Порівняння прогнозованих та реальних значень", padding=10)
        frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Введення реальних значень
        real_frame = ttk.LabelFrame(frame, text="Реальні значення (4 наступні квартали)", padding=10)
        real_frame.pack(fill='x', pady=10)
        
        instruction = ttk.Label(real_frame, text="Введіть реальні значення для порівняння з прогнозом:",
                              font=('Arial', 10))
        instruction.pack(anchor='w', pady=(0, 10))
        
        self.real_quarter_entries = {}
        
        for i in range(5, 9):
            row_frame = ttk.Frame(real_frame)
            row_frame.pack(fill='x', pady=5)
            
            label = ttk.Label(row_frame, text=f"Квартал {i} (Q{i}):", width=20)
            label.pack(side='left', padx=5)
            
            entry = ttk.Entry(row_frame, width=15)
            entry.pack(side='left', padx=5)
            entry.insert(0, str(round(0.40 + (i-4) * 0.04, 4)))
            
            self.real_quarter_entries[i] = entry
        
        # Параметри порогу
        threshold_frame = ttk.LabelFrame(frame, text="Обчислення порогу відхилення", padding=10)
        threshold_frame.pack(fill='x', pady=10)
        
        method_row = ttk.Frame(threshold_frame)
        method_row.pack(fill='x', pady=5)
        
        ttk.Label(method_row, text="Метод обчислення порогу:").pack(side='left', padx=5)
        self.threshold_method = ttk.Combobox(method_row, 
                                            values=['std', 'range', 'percent'],
                                            state='readonly', width=15)
        self.threshold_method.set('std')
        self.threshold_method.pack(side='left', padx=5)
        
        factor_row = ttk.Frame(threshold_frame)
        factor_row.pack(fill='x', pady=5)
        
        ttk.Label(factor_row, text="Множник порогу:").pack(side='left', padx=5)
        self.threshold_factor = ttk.Entry(factor_row, width=10)
        self.threshold_factor.insert(0, "1.5")
        self.threshold_factor.pack(side='left', padx=5)
        
        # Кнопки аналізу
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill='x', pady=10)
        
        btn_calculate_threshold = ttk.Button(btn_frame, text="Обчислити поріг",
                                            command=self.calculate_threshold)
        btn_calculate_threshold.pack(side='left', padx=5)
        
        btn_compare = ttk.Button(btn_frame, text="Порівняти тренди",
                                command=self.compare_trends)
        btn_compare.pack(side='left', padx=5)
        
        btn_visualize = ttk.Button(btn_frame, text="Побудувати графік порівняння",
                                  command=self.visualize_trend_comparison)
        btn_visualize.pack(side='left', padx=5)
        
        # Область для результатів
        self.trends_results_text = scrolledtext.ScrolledText(frame, height=20, width=100)
        self.trends_results_text.pack(pady=10, fill='both', expand=True)
        
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
            report += "Дата аналізу: 2025-11-11\n"
            report += "Підприємство: Coop - кооперативна роздрібна мережа\n"
            report += "═" * 80 + "\n"
            
            self.results_text.insert(1.0, report)
            
            messagebox.showinfo("Успіх", 
                              f"Узагальнений інтегральний показник впливу обчислено!\n\nIв = {integral_indicator:.6f}")
            
        except Exception as e:
            messagebox.showerror("Помилка", f"Помилка при обчисленні: {str(e)}")
    
    def calculate_for_quarter(self, quarter):
        """Автоматичне обчислення інтегрального показника для певного кварталу"""
        try:
            if self.group_priorities is None or not self.factor_priorities:
                messagebox.showerror("Помилка", 
                                   "Спочатку обчисліть вектори пріоритетів!")
                return
            
            # Обчислення інтегрального показника
            integral_indicator, _ = self.analyzer.calculate_integral_indicator(
                self.group_priorities, self.factor_priorities
            )
            
            # Додавання невеликої варіації для різних кварталів (симуляція)
            variation = np.random.uniform(-0.05, 0.05)
            quarter_value = integral_indicator + variation
            
            # Оновлення поля введення
            self.quarter_entries[quarter].delete(0, tk.END)
            self.quarter_entries[quarter].insert(0, str(round(quarter_value, 4)))
            
            messagebox.showinfo("Успіх", 
                              f"Інтегральний показник для Q{quarter} обчислено: {quarter_value:.4f}")
            
        except Exception as e:
            messagebox.showerror("Помилка", f"Помилка при обчисленні: {str(e)}")
    
    def save_quarterly_data(self):
        """Збереження квартальних даних"""
        try:
            quarterly_values = []
            
            for i in range(1, 5):
                value = float(self.quarter_entries[i].get())
                quarterly_values.append(value)
            
            self.quarterly_indicators = quarterly_values
            self.quarterly_analyzer.add_quarterly_data(quarterly_values)
            
            # Виведення результатів
            self.quarterly_results_text.delete(1.0, tk.END)
            
            result = "=" * 70 + "\n"
            result += "КВАРТАЛЬНІ ДАНІ ЗБЕРЕЖЕНО\n"
            result += "=" * 70 + "\n\n"
            
            result += "Узагальнений інтегральний показник впливу по кварталах:\n\n"
            
            for i, value in enumerate(quarterly_values, 1):
                result += f"   Q{i}: {value:.6f}\n"
            
            result += f"\nСереднє значення: {np.mean(quarterly_values):.6f}\n"
            result += f"Мінімальне значення: {np.min(quarterly_values):.6f}\n"
            result += f"Максимальне значення: {np.max(quarterly_values):.6f}\n"
            result += f"Стандартне відхилення: {np.std(quarterly_values):.6f}\n"
            
            # Визначення тренду
            if len(quarterly_values) >= 2:
                trend = "зростання" if quarterly_values[-1] > quarterly_values[0] else "спадання"
                result += f"\nТренд: {trend}\n"
            
            result += "\n" + "=" * 70 + "\n"
            
            self.quarterly_results_text.insert(1.0, result)
            
            messagebox.showinfo("Успіх", "Квартальні дані збережено успішно!")
            
        except ValueError:
            messagebox.showerror("Помилка", "Будь ласка, введіть коректні числові значення!")
        except Exception as e:
            messagebox.showerror("Помилка", f"Помилка при збереженні даних: {str(e)}")
    
    def visualize_quarterly_data(self):
        """Візуалізація квартальних даних"""
        try:
            if not self.quarterly_indicators:
                messagebox.showerror("Помилка", "Спочатку збережіть квартальні дані!")
                return
            
            # Створення вікна для графіка
            viz_window = tk.Toplevel(self.root)
            viz_window.title("Графік квартальних даних")
            viz_window.geometry("900x600")
            
            fig = Figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
            
            quarters = [f'Q{i}' for i in range(1, 5)]
            values = self.quarterly_indicators
            
            # Графік
            ax.plot(quarters, values, marker='o', linewidth=2, markersize=10, 
                   color='#2E86AB', label='Інтегральний показник')
            ax.fill_between(range(len(quarters)), values, alpha=0.3, color='#2E86AB')
            
            # Додавання значень на точках
            for i, (q, v) in enumerate(zip(quarters, values)):
                ax.annotate(f'{v:.4f}', xy=(i, v), textcoords="offset points",
                           xytext=(0, 10), ha='center', fontsize=10, fontweight='bold')
            
            # Лінія тренду
            x_numeric = np.array(range(len(quarters)))
            z = np.polyfit(x_numeric, values, 1)
            p = np.poly1d(z)
            ax.plot(quarters, p(x_numeric), "--", color='red', alpha=0.8, 
                   linewidth=2, label=f'Тренд (y={z[0]:.4f}x+{z[1]:.4f})')
            
            # Середнє значення
            mean_val = np.mean(values)
            ax.axhline(y=mean_val, color='green', linestyle=':', linewidth=2, 
                      label=f'Середнє ({mean_val:.4f})')
            
            ax.set_xlabel('Квартал', fontsize=12, fontweight='bold')
            ax.set_ylabel('Інтегральний показник впливу (Iв)', fontsize=12, fontweight='bold')
            ax.set_title('Зміна узагальненого інтегрального показника впливу по кварталах', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='best', fontsize=10)
            
            fig.tight_layout()
            
            canvas = FigureCanvasTkAgg(fig, master=viz_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            
        except Exception as e:
            messagebox.showerror("Помилка", f"Помилка при побудові графіка: {str(e)}")
    
    def perform_forecast(self):
        """Виконання прогнозування"""
        try:
            if not self.quarterly_indicators:
                messagebox.showerror("Помилка", "Спочатку збережіть квартальні дані!")
                return
            
            method = self.forecast_method.get()
            
            # Виконання прогнозування в залежності від методу
            if method == "linear":
                predictions = self.quarterly_analyzer.forecast_linear_regression(n_future=4)
            elif method == "polynomial":
                degree = int(self.poly_degree.get())
                predictions = self.quarterly_analyzer.forecast_polynomial_regression(n_future=4, degree=degree)
            elif method == "exponential":
                alpha = float(self.alpha_entry.get())
                predictions = self.quarterly_analyzer.forecast_exponential_smoothing(n_future=4, alpha=alpha)
            elif method == "moving_average":
                window = int(self.window_size.get())
                predictions = self.quarterly_analyzer.forecast_moving_average(n_future=4, window=window)
            else:
                messagebox.showerror("Помилка", "Невідомий метод прогнозування!")
                return
            
            # Виведення результатів
            self.forecast_results_text.delete(1.0, tk.END)
            
            result = "=" * 70 + "\n"
            result += "РЕЗУЛЬТАТИ ПРОГНОЗУВАННЯ\n"
            result += "=" * 70 + "\n\n"
            
            method_names = {
                "linear": "Лінійна регресія",
                "polynomial": "Поліноміальна регресія",
                "exponential": "Експоненціальне згладжування",
                "moving_average": "Ковзне середнє"
            }
            
            result += f"Метод прогнозування: {method_names[method]}\n\n"
            
            result += "ІСТОРИЧНІ ДАНІ (4 квартали):\n"
            result += "-" * 70 + "\n"
            for i, value in enumerate(self.quarterly_indicators, 1):
                result += f"   Q{i}: {value:.6f}\n"
            
            result += "\nПРОГНОЗОВАНІ ЗНАЧЕННЯ (4 наступні квартали):\n"
            result += "-" * 70 + "\n"
            for i, value in enumerate(predictions, 5):
                result += f"   Q{i}: {value:.6f}\n"
            
            result += "\nСТАТИСТИКА:\n"
            result += "-" * 70 + "\n"
            result += f"   Середнє (історичні дані): {np.mean(self.quarterly_indicators):.6f}\n"
            result += f"   Середнє (прогноз): {np.mean(predictions):.6f}\n"
            result += f"   Зміна середнього: {(np.mean(predictions) - np.mean(self.quarterly_indicators)):.6f}\n"
            
            # Визначення тренду прогнозу
            if predictions[-1] > predictions[0]:
                trend = "Зростання"
            elif predictions[-1] < predictions[0]:
                trend = "Спадання"
            else:
                trend = "Стабільність"
            
            result += f"   Прогнозований тренд: {trend}\n"
            
            result += "\n" + "=" * 70 + "\n"
            
            self.forecast_results_text.insert(1.0, result)
            
            messagebox.showinfo("Успіх", "Прогнозування виконано успішно!")
            
        except ValueError as e:
            messagebox.showerror("Помилка", f"Помилка у параметрах: {str(e)}")
        except Exception as e:
            messagebox.showerror("Помилка", f"Помилка при прогнозуванні: {str(e)}")
    
    def visualize_forecast(self):
        """Візуалізація прогнозу"""
        try:
            if not self.quarterly_indicators or not self.quarterly_analyzer.predicted_data:
                messagebox.showerror("Помилка", "Спочатку виконайте прогнозування!")
                return
            
            # Створення вікна для графіка
            viz_window = tk.Toplevel(self.root)
            viz_window.title("Графік прогнозування")
            viz_window.geometry("1000x600")
            
            fig = Figure(figsize=(12, 6))
            ax = fig.add_subplot(111)
            
            # Історичні дані
            hist_quarters = [f'Q{i}' for i in range(1, 5)]
            hist_values = self.quarterly_indicators
            
            # Прогнозовані дані
            pred_quarters = [f'Q{i}' for i in range(5, 9)]
            pred_values = self.quarterly_analyzer.predicted_data
            
            # Всі квартали для х-осі
            all_quarters = hist_quarters + pred_quarters
            
            # Графік історичних даних
            ax.plot(range(len(hist_quarters)), hist_values, marker='o', linewidth=2.5, 
                   markersize=10, color='#2E86AB', label='Історичні дані', zorder=3)
            
            # Графік прогнозу
            ax.plot(range(len(hist_quarters)-1, len(all_quarters)), 
                   [hist_values[-1]] + pred_values, 
                   marker='s', linewidth=2.5, markersize=10, 
                   color='#F18F01', linestyle='--', label='Прогноз', zorder=3)
            
            # Заповнення областей
            ax.fill_between(range(len(hist_quarters)), hist_values, alpha=0.3, color='#2E86AB')
            ax.fill_between(range(len(hist_quarters)-1, len(all_quarters)), 
                           [hist_values[-1]] + pred_values, alpha=0.3, color='#F18F01')
            
            # Додавання значень
            for i, v in enumerate(hist_values):
                ax.annotate(f'{v:.4f}', xy=(i, v), textcoords="offset points",
                           xytext=(0, 10), ha='center', fontsize=9, fontweight='bold')
            
            for i, v in enumerate(pred_values):
                ax.annotate(f'{v:.4f}', xy=(len(hist_quarters)+i, v), 
                           textcoords="offset points", xytext=(0, 10), 
                           ha='center', fontsize=9, fontweight='bold', color='#F18F01')
            
            # Вертикальна лінія розділення
            ax.axvline(x=len(hist_quarters)-0.5, color='gray', linestyle=':', 
                      linewidth=2, alpha=0.7, label='Межа прогнозу')
            
            # Налаштування осей
            ax.set_xticks(range(len(all_quarters)))
            ax.set_xticklabels(all_quarters)
            ax.set_xlabel('Квартал', fontsize=12, fontweight='bold')
            ax.set_ylabel('Інтегральний показник впливу (Iв)', fontsize=12, fontweight='bold')
            ax.set_title('Прогноз узагальненого інтегрального показника впливу', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='best', fontsize=11)
            
            fig.tight_layout()
            
            canvas = FigureCanvasTkAgg(fig, master=viz_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            
        except Exception as e:
            messagebox.showerror("Помилка", f"Помилка при побудові графіка: {str(e)}")
    
    def calculate_threshold(self):
        """Обчислення порогового значення"""
        try:
            if not self.quarterly_indicators:
                messagebox.showerror("Помилка", "Спочатку збережіть квартальні дані!")
                return
            
            method = self.threshold_method.get()
            factor = float(self.threshold_factor.get())
            
            threshold = self.quarterly_analyzer.calculate_threshold(method=method, factor=factor)
            
            # Виведення результату
            self.trends_results_text.delete(1.0, tk.END)
            
            result = "=" * 70 + "\n"
            result += "ОБЧИСЛЕННЯ ПОРОГУ ВІДХИЛЕННЯ\n"
            result += "=" * 70 + "\n\n"
            
            method_names = {
                'std': 'Стандартне відхилення',
                'range': 'Діапазон значень',
                'percent': 'Відсоток від середнього'
            }
            
            result += f"Метод: {method_names.get(method, method)}\n"
            result += f"Множник: {factor}\n\n"
            
            result += "СТАТИСТИКА ІСТОРИЧНИХ ДАНИХ:\n"
            result += "-" * 70 + "\n"
            result += f"   Середнє значення: {np.mean(self.quarterly_indicators):.6f}\n"
            result += f"   Стандартне відхилення: {np.std(self.quarterly_indicators):.6f}\n"
            result += f"   Мінімум: {np.min(self.quarterly_indicators):.6f}\n"
            result += f"   Максимум: {np.max(self.quarterly_indicators):.6f}\n"
            result += f"   Діапазон: {np.max(self.quarterly_indicators) - np.min(self.quarterly_indicators):.6f}\n"
            
            result += f"\nПОРОГОВЕ ЗНАЧЕННЯ:\n"
            result += "-" * 70 + "\n"
            result += f"   Поріг (δ): {threshold:.6f}\n\n"
            
            result += "ІНТЕРПРЕТАЦІЯ:\n"
            result += "-" * 70 + "\n"
            result += f"   Якщо |Ivп(tj) - Ivн(tj)| ≤ {threshold:.6f}, тренд вважається СТАБІЛЬНИМ\n"
            result += f"   Якщо Ivп(tj) > Ivн(tj) + {threshold:.6f}, спостерігається ЗРОСТАННЯ\n"
            result += f"   Якщо Ivп(tj) < Ivн(tj) - {threshold:.6f}, спостерігається СПАДАННЯ\n"
            
            result += "\n" + "=" * 70 + "\n"
            
            self.trends_results_text.insert(1.0, result)
            
            messagebox.showinfo("Успіх", f"Поріг відхилення обчислено: δ = {threshold:.6f}")
            
        except ValueError:
            messagebox.showerror("Помилка", "Будь ласка, введіть коректні числові значення!")
        except Exception as e:
            messagebox.showerror("Помилка", f"Помилка при обчисленні порогу: {str(e)}")
    
    def compare_trends(self):
        """Порівняння прогнозованих та реальних трендів"""
        try:
            if not self.quarterly_analyzer.predicted_data:
                messagebox.showerror("Помилка", "Спочатку виконайте прогнозування!")
                return
            
            if self.quarterly_analyzer.threshold == 0:
                messagebox.showerror("Помилка", "Спочатку обчисліть поріг відхилення!")
                return
            
            # Отримання реальних значень
            real_values = []
            for i in range(5, 9):
                value = float(self.real_quarter_entries[i].get())
                real_values.append(value)
            
            self.real_future_values = real_values
            
            # Порівняння
            deviations, trends = self.quarterly_analyzer.compare_forecast_with_real(real_values)
            
            # Формування звіту
            self.trends_results_text.delete(1.0, tk.END)
            
            result = "╔" + "═" * 78 + "╗\n"
            result += "║" + " " * 22 + "ПОРІВНЯННЯ ТРЕНДІВ" + " " * 39 + "║\n"
            result += "║" + " " * 15 + "Прогнозовані vs Реальні значення" + " " * 31 + "║\n"
            result += "╚" + "═" * 78 + "╝\n\n"
            
            result += "┌" + "─" * 78 + "┐\n"
            result += "│ ПОРІВНЯЛЬНА ТАБЛИЦЯ" + " " * 58 + "│\n"
            result += "└" + "─" * 78 + "┘\n\n"
            
            result += f"{'Квартал':<10} {'Прогноз (Ivп)':<18} {'Реальне (Ivн)':<18} {'Відхилення':<15} {'Тренд':<20}\n"
            result += "-" * 80 + "\n"
            
            predicted = self.quarterly_analyzer.predicted_data
            threshold = self.quarterly_analyzer.threshold
            
            for i in range(4):
                quarter = f"Q{i+5}"
                pred_val = predicted[i]
                real_val = real_values[i]
                deviation = deviations[i]
                trend = trends[i]
                
                # Символ для тренду
                if trend == "Стабільний":
                    trend_symbol = "→"
                elif trend == "Зростання":
                    trend_symbol = "↑"
                else:
                    trend_symbol = "↓"
                
                result += f"{quarter:<10} {pred_val:<18.6f} {real_val:<18.6f} "
                result += f"{deviation:<15.6f} {trend_symbol} {trend:<18}\n"
            
            result += "\n┌" + "─" * 78 + "┐\n"
            result += "│ СТАТИСТИЧНИЙ АНАЛІЗ" + " " * 58 + "│\n"
            result += "└" + "─" * 78 + "┘\n\n"
            
            result += f"Поріг відхилення (δ): {threshold:.6f}\n\n"
            
            # Точність прогнозу
            mape = np.mean([abs((r - p) / r) for r, p in zip(real_values, predicted)]) * 100
            result += f"Середня абсолютна відсоткова помилка (MAPE): {mape:.2f}%\n"
            
            mae = np.mean(deviations)
            result += f"Середнє абсолютне відхилення (MAE): {mae:.6f}\n"
            
            rmse = np.sqrt(np.mean([(r - p)**2 for r, p in zip(real_values, predicted)]))
            result += f"Середньоквадратична помилка (RMSE): {rmse:.6f}\n"
            
            result += "\n┌" + "─" * 78 + "┐\n"
            result += "│ АНАЛІЗ ТРЕНДІВ" + " " * 63 + "│\n"
            result += "└" + "─" * 78 + "┘\n\n"
            
            # Підрахунок трендів
            trend_counts = {"Стабільний": 0, "Зростання": 0, "Спадання": 0}
            for trend in trends:
                trend_counts[trend] += 1
            
            result += "Розподіл трендів:\n"
            for trend_type, count in trend_counts.items():
                percentage = (count / len(trends)) * 100
                bar_length = int(percentage / 5)
                bar = "█" * bar_length
                result += f"   {trend_type:<15}: {count} ({percentage:.1f}%) {bar}\n"
            
            # Загальний висновок
            result += "\n┌" + "─" * 78 + "┐\n"
            result += "│ ВИСНОВКИ" + " " * 69 + "│\n"
            result += "└" + "─" * 78 + "┘\n\n"
            
            if mape < 10:
                accuracy = "Відмінна"
            elif mape < 20:
                accuracy = "Добра"
            elif mape < 30:
                accuracy = "Задовільна"
            else:
                accuracy = "Низька"
            
            result += f"■ Точність прогнозу: {accuracy} (MAPE = {mape:.2f}%)\n"
            
            dominant_trend = max(trend_counts.items(), key=lambda x: x[1])[0]
            result += f"■ Домінуючий тренд: {dominant_trend}\n"
            
            # Порівняння середніх
            avg_pred = np.mean(predicted)
            avg_real = np.mean(real_values)
            diff_avg = avg_real - avg_pred
            
            if abs(diff_avg) <= threshold:
                result += f"■ Прогноз добре узгоджується з реальними даними\n"
            elif diff_avg > threshold:
                result += f"■ Реальні значення перевищують прогноз на {diff_avg:.6f}\n"
            else:
                result += f"■ Реальні значення нижче прогнозу на {abs(diff_avg):.6f}\n"
            
            result += "\n" + "═" * 80 + "\n"
            result += "Дата аналізу: 2025-11-11\n"
            result += "═" * 80 + "\n"
            
            self.trends_results_text.insert(1.0, result)
            
            messagebox.showinfo("Успіх", "Порівняння трендів виконано успішно!")
            
        except ValueError:
            messagebox.showerror("Помилка", "Будь ласка, введіть коректні числові значення!")
        except Exception as e:
            messagebox.showerror("Помилка", f"Помилка при порівнянні: {str(e)}")
    
    def visualize_trend_comparison(self):
        """Візуалізація порівняння трендів"""
        try:
            if not self.real_future_values or not self.quarterly_analyzer.predicted_data:
                messagebox.showerror("Помилка", "Спочатку виконайте порівняння трендів!")
                return
            
            # Створення вікна для графіка
            viz_window = tk.Toplevel(self.root)
            viz_window.title("Порівняння прогнозу та реальних значень")
            viz_window.geometry("1200x700")
            
            # Створення головної фігури з підграфіками
            fig = Figure(figsize=(14, 8))
            
            # Графік 1: Порівняння всіх значень
            ax1 = fig.add_subplot(2, 2, (1, 2))
            
            # Дані
            hist_quarters = [f'Q{i}' for i in range(1, 5)]
            hist_values = self.quarterly_indicators
            pred_quarters = [f'Q{i}' for i in range(5, 9)]
            pred_values = self.quarterly_analyzer.predicted_data
            real_values = self.real_future_values
            all_quarters = hist_quarters + pred_quarters
            
            # Графіки
            x_hist = range(len(hist_quarters))
            x_pred = range(len(hist_quarters)-1, len(all_quarters))
            
            # Історичні дані
            ax1.plot(x_hist, hist_values, marker='o', linewidth=2.5, markersize=10,
                    color='#2E86AB', label='Історичні дані', zorder=3)
            
            # Прогноз
            ax1.plot(x_pred, [hist_values[-1]] + pred_values, marker='s', 
                    linewidth=2.5, markersize=10, color='#F18F01', 
                    linestyle='--', label='Прогноз (Ivп)', zorder=3)
            
            # Реальні значення
            ax1.plot(x_pred, [hist_values[-1]] + real_values, marker='D', 
                    linewidth=2.5, markersize=10, color='#06A77D', 
                    linestyle='-', label='Реальні дані (Ivн)', zorder=3)
            
            # Поріг
            if self.quarterly_analyzer.threshold > 0:
                threshold = self.quarterly_analyzer.threshold
                for i, (p, r) in enumerate(zip(pred_values, real_values), len(hist_quarters)):
                    ax1.fill_between([i-0.3, i+0.3], [p-threshold]*2, [p+threshold]*2,
                                    alpha=0.2, color='yellow', label='Поріг' if i == len(hist_quarters) else '')
            
            # Вертикальна лінія
            ax1.axvline(x=len(hist_quarters)-0.5, color='gray', linestyle=':', 
                       linewidth=2, alpha=0.7, label='Межа прогнозу')
            
            ax1.set_xticks(range(len(all_quarters)))
            ax1.set_xticklabels(all_quarters)
            ax1.set_xlabel('Квартал', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Інтегральний показник (Iв)', fontsize=12, fontweight='bold')
            ax1.set_title('Порівняння прогнозованих та реальних значень', 
                         fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3, linestyle='--')
            ax1.legend(loc='best', fontsize=10)
            
            # Графік 2: Відхилення
            ax2 = fig.add_subplot(2, 2, 3)
            
            deviations = [abs(r - p) for r, p in zip(real_values, pred_values)]
            colors_dev = ['green' if d <= self.quarterly_analyzer.threshold else 'red' 
                         for d in deviations]
            
            bars = ax2.bar(pred_quarters, deviations, color=colors_dev, alpha=0.7, edgecolor='black')
            
            if self.quarterly_analyzer.threshold > 0:
                ax2.axhline(y=self.quarterly_analyzer.threshold, color='orange', 
                           linestyle='--', linewidth=2, label=f'Поріг ({self.quarterly_analyzer.threshold:.4f})')
            
            # Додавання значень
            for bar, dev in zip(bars, deviations):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{dev:.4f}', ha='center', va='bottom', fontsize=9)
            
            ax2.set_xlabel('Квартал', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Абсолютне відхилення', fontsize=11, fontweight='bold')
            ax2.set_title('Відхилення прогнозу від реальних значень', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
            ax2.legend(loc='best', fontsize=9)
            
            # Графік 3: Відсоткові помилки
            ax3 = fig.add_subplot(2, 2, 4)
            
            percent_errors = [abs((r - p) / r * 100) for r, p in zip(real_values, pred_values)]
            
            bars = ax3.bar(pred_quarters, percent_errors, color='#9B59B6', alpha=0.7, edgecolor='black')
            
            avg_error = np.mean(percent_errors)
            ax3.axhline(y=avg_error, color='red', linestyle='--', linewidth=2, 
                       label=f'Середнє ({avg_error:.2f}%)')
            
            # Додавання значень
            for bar, err in zip(bars, percent_errors):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{err:.1f}%', ha='center', va='bottom', fontsize=9)
            
            ax3.set_xlabel('Квартал', fontsize=11, fontweight='bold')
            ax3.set_ylabel('Відсоткова помилка (%)', fontsize=11, fontweight='bold')
            ax3.set_title('Відсоткові помилки прогнозування (MAPE)', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
            ax3.legend(loc='best', fontsize=9)
            
            fig.tight_layout(pad=2.0)
            
            canvas = FigureCanvasTkAgg(fig, master=viz_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            
        except Exception as e:
            messagebox.showerror("Помилка", f"Помилка при побудові графіка: {str(e)}")
    
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