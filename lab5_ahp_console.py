#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Лабораторна робота №5
Імітаційна модель визначення пріоритетності інноваційних проектів
Метод аналізу ієрархій (МАІ)
Консольна версія
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Для збереження графіків без GUI
import matplotlib.pyplot as plt
from datetime import datetime

class AHPModel:
    """Клас для реалізації методу аналізу ієрархій"""
    
    def __init__(self):
        self.criteria_groups = ['Економічні', 'Технічні', 'Виробничі', 'Екологічні', 'Організаційні']
        self.num_criteria = 4  # Кількість критеріїв в кожній групі
        self.num_projects = 3
        
    def create_pairwise_matrix(self, weights):
        """Створення матриці попарних порівнянь за формулою aij = wi/wj"""
        n = len(weights)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if weights[j] != 0:
                    matrix[i, j] = weights[i] / weights[j]
                else:
                    matrix[i, j] = 1.0
        return matrix
    
    def calculate_eigenvector(self, matrix):
        """Обчислення власного вектора за формулою aiв = (∏j aij)^(1/n)"""
        n = matrix.shape[0]
        eigenvector = np.zeros(n)
        for i in range(n):
            product = np.prod(matrix[i, :])
            eigenvector[i] = product ** (1.0 / n)
        return eigenvector
    
    def calculate_priority_vector(self, eigenvector):
        """Обчислення вектора пріоритетів: aip = aiв / Σaiв"""
        total = np.sum(eigenvector)
        if total == 0:
            return np.zeros_like(eigenvector)
        priority_vector = eigenvector / total
        return priority_vector
    
    def calculate_consistency_index(self, matrix, priority_vector):
        """Обчислення індексу узгодженості"""
        n = matrix.shape[0]
        weighted_sum = np.dot(matrix, priority_vector)
        lambda_max = np.sum(weighted_sum / priority_vector) / n
        CI = (lambda_max - n) / (n - 1) if n > 1 else 0
        return CI, lambda_max
    
    def calculate_consistency_ratio(self, CI, n):
        """Обчислення відношення узгодженості CR"""
        RI = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
        if n in RI and RI[n] != 0:
            CR = CI / RI[n]
        else:
            CR = 0
        return CR
    
    def calculate_global_priorities(self, priority_matrix, criteria_priorities):
        """Обчислення глобальних пріоритетів G = P × Ap"""
        global_priorities = np.dot(priority_matrix, criteria_priorities)
        return global_priorities


def run_ahp_analysis():
    """Основна функція виконання аналізу"""
    
    print("="*80)
    print("ЛАБОРАТОРНА РОБОТА №5")
    print("Імітаційна модель визначення пріоритетності інноваційних проектів")
    print("Метод аналізу ієрархій (МАІ)")
    print("="*80)
    print()
    
    # Створення моделі
    ahp = AHPModel()
    
    # ВХІДНІ ДАНІ
    print("Завантаження вхідних даних...")
    
    # Ваги груп критеріїв (2-й рівень)
    group_weights = np.array([7, 5, 4, 6, 3])
    
    # Ваги для проектів (3-й рівень)
    project1_weights = [
        [8, 7, 9, 6],  # Економічні
        [5, 6, 7, 5],  # Технічні
        [8, 7, 9, 6],  # Виробничі
        [9, 8, 7, 8],  # Екологічні
        [5, 6, 5, 4]   # Організаційні
    ]
    
    project2_weights = [
        [6, 7, 5, 8],
        [8, 9, 7, 8],
        [5, 6, 4, 6],
        [4, 5, 6, 4],
        [7, 8, 7, 6]
    ]
    
    project3_weights = [
        [4, 5, 3, 5],
        [6, 5, 6, 4],
        [4, 5, 3, 5],
        [6, 7, 5, 6],
        [4, 5, 6, 5]
    ]
    
    all_projects = [project1_weights, project2_weights, project3_weights]
    
    # Відкриваємо файл для запису результатів
    output = []
    
    output.append("="*80)
    output.append("ЛАБОРАТОРНА РОБОТА №5")
    output.append("Розробка та дослідження імітаційної моделі для вибору інноваційних проектів")
    output.append("Метод аналізу ієрархій (МАІ)")
    output.append(f"Дата виконання: {datetime.now().strftime('%d.%m.%Y %H:%M')}")
    output.append("="*80)
    output.append("")
    
    # КРОК 3: Матриця попарних порівнянь для 2-го рівня
    output.append("="*80)
    output.append("КРОК 3: ФОРМУВАННЯ МАТРИЦІ ПОПАРНИХ ПОРІВНЯНЬ ДЛЯ 2-ГО РІВНЯ")
    output.append("="*80)
    output.append("")
    
    output.append("Вхідні ваги груп критеріїв:")
    for i, (group, weight) in enumerate(zip(ahp.criteria_groups, group_weights)):
        output.append(f"  {group}: w = {weight}")
    output.append("")
    
    output.append("Матриця попарних порівнянь A (за формулою aij = wi/wj):")
    output.append("")
    
    pairwise_matrix_level2 = ahp.create_pairwise_matrix(group_weights)
    
    # Форматований вивід матриці
    header = "        " + "".join([f"{g[:4]:>10}" for g in ahp.criteria_groups])
    output.append(header)
    for i, group in enumerate(ahp.criteria_groups):
        row = f"{group[:4]:>6}  "
        for j in range(len(ahp.criteria_groups)):
            row += f"{pairwise_matrix_level2[i,j]:>10.3f}"
        output.append(row)
    output.append("")
    
    # Обчислення власного вектора
    eigenvector = ahp.calculate_eigenvector(pairwise_matrix_level2)
    output.append("Власний вектор Aв (формула: aiв = (∏j aij)^(1/n)):")
    for i, (group, value) in enumerate(zip(ahp.criteria_groups, eigenvector)):
        output.append(f"  {group}: {value:.4f}")
    output.append("")
    
    # Обчислення вектора пріоритетів
    priority_vector_level2 = ahp.calculate_priority_vector(eigenvector)
    output.append("Вектор пріоритетів Aр (формула: aip = aiв / Σaiв):")
    for i, (group, value) in enumerate(zip(ahp.criteria_groups, priority_vector_level2)):
        output.append(f"  {group}: {value:.4f} ({value*100:.2f}%)")
    output.append("")
    
    # Перевірка узгодженості
    CI, lambda_max = ahp.calculate_consistency_index(pairwise_matrix_level2, priority_vector_level2)
    CR = ahp.calculate_consistency_ratio(CI, len(ahp.criteria_groups))
    
    output.append("Перевірка узгодженості:")
    output.append(f"  λmax = {lambda_max:.4f}")
    output.append(f"  Індекс узгодженості (CI) = {CI:.4f}")
    output.append(f"  Відношення узгодженості (CR) = {CR:.4f}")
    if CR < 0.1:
        output.append("  ✓ Матриця узгоджена (CR < 0.1)")
    else:
        output.append("  ✗ УВАГА: Матриця неузгоджена (CR >= 0.1)")
    output.append("")
    
    print("✓ Крок 3 завершено")
    
    # КРОК 4: Матриці для кожного проекту
    output.append("="*80)
    output.append("КРОК 4: ФОРМУВАННЯ МАТРИЦЬ ДЛЯ КОЖНОГО ПРОЕКТУ (3-Й РІВЕНЬ)")
    output.append("="*80)
    output.append("")
    
    output.append("4.1. Вхідні дані по проектах:")
    output.append("")
    
    for p in range(ahp.num_projects):
        output.append(f"ПРОЕКТ {p+1}:")
        for g, group in enumerate(ahp.criteria_groups):
            criteria_weights = all_projects[p][g]
            output.append(f"  {group}: {criteria_weights}")
        output.append("")
    
    output.append("4.2. Попарне порівняння проектів за кожною групою критеріїв:")
    output.append("")
    
    project_priorities = []
    
    # Для кожної групи критеріїв порівнюємо проекти між собою
    for g, group in enumerate(ahp.criteria_groups):
        output.append(f"{'─'*80}")
        output.append(f"ГРУПА КРИТЕРІЇВ: {group}")
        output.append(f"{'─'*80}")
        output.append("")
        
        # Збираємо середні ваги критеріїв для всіх проектів у цій групі
        project_avg_weights = []
        for p in range(ahp.num_projects):
            criteria_weights = np.array(all_projects[p][g])
            avg_weight = np.mean(criteria_weights)
            project_avg_weights.append(avg_weight)
            output.append(f"Проект {p+1}: {criteria_weights} → середнє = {avg_weight:.4f}")
        
        output.append("")
        
        # Створюємо матрицю попарних порівнянь проектів за цією групою
        project_comparison_matrix = ahp.create_pairwise_matrix(np.array(project_avg_weights))
        
        output.append("Матриця попарних порівнянь проектів:")
        header = "         " + "".join([f"Проект{i+1:>7}" for i in range(ahp.num_projects)])
        output.append(header)
        for p in range(ahp.num_projects):
            row = f"Проект{p+1}  "
            for j in range(ahp.num_projects):
                row += f"{project_comparison_matrix[p,j]:>10.3f}"
            output.append(row)
        output.append("")
        
        # Обчислюємо пріоритети проектів для цієї групи
        eigenvector = ahp.calculate_eigenvector(project_comparison_matrix)
        group_priority_vector = ahp.calculate_priority_vector(eigenvector)
        
        output.append(f"Вектор пріоритетів проектів для групи '{group}':")
        for p, priority in enumerate(group_priority_vector):
            output.append(f"  Проект {p+1}: {priority:.4f}")
        output.append("")
        
        # Зберігаємо пріоритети для цієї групи
        for p in range(ahp.num_projects):
            if len(project_priorities) <= p:
                project_priorities.append([])
            project_priorities[p].append(group_priority_vector[p])
    
    print("✓ Крок 4 завершено")
    
    # КРОК 5: Глобальні пріоритети
    output.append("="*80)
    output.append("КРОК 5: ОБЧИСЛЕННЯ ГЛОБАЛЬНИХ ПРІОРИТЕТІВ")
    output.append("="*80)
    output.append("")
    
    # Матриця пріоритетів (проекти × групи)
    priority_matrix = np.array(project_priorities)
    
    output.append("Матриця пріоритетів P (проекти × групи критеріїв):")
    output.append("")
    header = "         " + "".join([f"{g[:4]:>12}" for g in ahp.criteria_groups])
    output.append(header)
    for p in range(ahp.num_projects):
        row = f"Проект{p+1}  "
        for g in range(len(ahp.criteria_groups)):
            row += f"{priority_matrix[p,g]:>12.4f}"
        output.append(row)
    output.append("")
    
    output.append("Вектор пріоритетів груп Aр:")
    for group, priority in zip(ahp.criteria_groups, priority_vector_level2):
        output.append(f"  {group}: {priority:.4f}")
    output.append("")
    
    # Обчислення G = P × Aр
    global_priorities = ahp.calculate_global_priorities(priority_matrix, priority_vector_level2)
    
    output.append("Вектор глобальних пріоритетів G = P × Aр:")
    output.append("")
    for p, priority in enumerate(global_priorities):
        output.append(f"  Проект {p+1}: G{p+1} = {priority:.4f} ({priority*100:.2f}%)")
    output.append("")
    
    print("✓ Крок 5 завершено")
    
    # КРОК 6: Вибір проекту
    output.append("="*80)
    output.append("КРОК 6: ВИБІР НАЙКРАЩОГО ПРОЕКТУ")
    output.append("="*80)
    output.append("")
    
    best_project = np.argmax(global_priorities) + 1
    best_priority = np.max(global_priorities)
    
    # Сортування проектів
    sorted_indices = np.argsort(global_priorities)[::-1]
    
    output.append("Ранжування проектів:")
    output.append("")
    for rank, idx in enumerate(sorted_indices, 1):
        priority = global_priorities[idx]
        line = f"  {rank}. Проект {idx+1}: {priority:.4f} ({priority*100:.2f}%)"
        if rank == 1:
            line += " ← РЕКОМЕНДОВАНО ✓"
        output.append(line)
    output.append("")
    
    output.append("="*80)
    output.append("ВИСНОВОК:")
    output.append(f"До першочергової реалізації рекомендується ПРОЕКТ {best_project}")
    output.append(f"з глобальним пріоритетом {best_priority:.4f} ({best_priority*100:.2f}%)")
    output.append("="*80)
    
    print("✓ Крок 6 завершено")
    
    # Збереження результатів у файл
    print("\nЗбереження результатів...")
    with open('/mnt/user-data/outputs/lab5_results.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(output))
    
    print("✓ Текстовий звіт збережено: lab5_results.txt")
    
    # Побудова графіків
    print("\nПобудова графіків...")
    
    fig = plt.figure(figsize=(14, 10))
    
    # Графік 1: Стовпчикова діаграма глобальних пріоритетів
    ax1 = plt.subplot(2, 3, 1)
    projects = [f'Проект {i+1}' for i in range(ahp.num_projects)]
    colors = ['#2ecc71' if i == sorted_indices[0] else '#3498db' 
             for i in range(ahp.num_projects)]
    
    bars = ax1.bar(projects, global_priorities * 100, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Глобальний пріоритет (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Порівняння проектів', fontsize=12, fontweight='bold', pad=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, max(global_priorities * 100) * 1.2)
    
    # Додавання значень на стовпчиках
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Графік 2: Кругова діаграма
    ax2 = plt.subplot(2, 3, 2)
    explode = [0.1 if i == sorted_indices[0] else 0 for i in range(ahp.num_projects)]
    
    wedges, texts, autotexts = ax2.pie(global_priorities, labels=projects, autopct='%1.2f%%',
           explode=explode, shadow=True, startangle=90, colors=colors)
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    ax2.set_title('Розподіл пріоритетів', fontsize=12, fontweight='bold', pad=10)
    
    # Графік 3: Пріоритети груп критеріїв
    ax3 = plt.subplot(2, 3, 3)
    groups_short = [g for g in ahp.criteria_groups]
    
    bars = ax3.barh(groups_short, priority_vector_level2 * 100, color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax3.set_xlabel('Пріоритет (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Ваги груп критеріїв', fontsize=12, fontweight='bold', pad=10)
    ax3.grid(axis='x', alpha=0.3, linestyle='--')
    ax3.set_xlim(0, max(priority_vector_level2 * 100) * 1.2)
    
    # Додавання значень
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax3.text(width, bar.get_y() + bar.get_height()/2.,
                f'{width:.2f}%', ha='left', va='center', fontsize=9, fontweight='bold', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Графік 4: Матриця пріоритетів проектів (теплова карта)
    ax4 = plt.subplot(2, 3, 4)
    
    im = ax4.imshow(priority_matrix.T, cmap='YlOrRd', aspect='auto')
    ax4.set_xticks(range(ahp.num_projects))
    ax4.set_xticklabels([f'Проект {i+1}' for i in range(ahp.num_projects)])
    ax4.set_yticks(range(len(ahp.criteria_groups)))
    ax4.set_yticklabels(ahp.criteria_groups, fontsize=9)
    ax4.set_title('Матриця пріоритетів проектів', fontsize=12, fontweight='bold', pad=10)
    
    # Додавання значень на теплову карту
    for i in range(priority_matrix.shape[1]):
        for j in range(priority_matrix.shape[0]):
            text = ax4.text(j, i, f'{priority_matrix[j, i]:.3f}',
                          ha="center", va="center", color="black", fontsize=8, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Пріоритет', rotation=270, labelpad=15, fontweight='bold')
    
    # Графік 5: Порівняльна діаграма по групах
    ax5 = plt.subplot(2, 3, 5)
    
    x = np.arange(len(ahp.criteria_groups))
    width = 0.25
    
    for i in range(ahp.num_projects):
        offset = (i - 1) * width
        bars = ax5.bar(x + offset, priority_matrix[i] * 100, width, 
                      label=f'Проект {i+1}', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax5.set_xlabel('Групи критеріїв', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Пріоритет (%)', fontsize=11, fontweight='bold')
    ax5.set_title('Порівняння проектів по групах', fontsize=12, fontweight='bold', pad=10)
    ax5.set_xticks(x)
    ax5.set_xticklabels([g[:8] for g in ahp.criteria_groups], rotation=45, ha='right')
    ax5.legend(loc='upper right', fontsize=9)
    ax5.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Графік 6: Радарна діаграма
    ax6 = plt.subplot(2, 3, 6, projection='polar')
    
    angles = np.linspace(0, 2 * np.pi, len(ahp.criteria_groups), endpoint=False).tolist()
    angles += angles[:1]  # Замикаємо коло
    
    for i in range(ahp.num_projects):
        values = priority_matrix[i].tolist()
        values += values[:1]  # Замикаємо коло
        ax6.plot(angles, values, 'o-', linewidth=2, label=f'Проект {i+1}', alpha=0.8)
        ax6.fill(angles, values, alpha=0.15)
    
    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels([g[:8] for g in ahp.criteria_groups], fontsize=9)
    ax6.set_ylim(0, max(priority_matrix.flatten()) * 1.2)
    ax6.set_title('Профіль проектів', fontsize=12, fontweight='bold', pad=20)
    ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/lab5_charts.png', dpi=300, bbox_inches='tight')
    
    print("✓ Графіки збережено: lab5_charts.png")
    
    # Виведення короткого резюме
    print("\n" + "="*80)
    print("РЕЗУЛЬТАТИ АНАЛІЗУ")
    print("="*80)
    print(f"\nГлобальні пріоритети:")
    for p, priority in enumerate(global_priorities):
        marker = " ✓ РЕКОМЕНДОВАНО" if p == sorted_indices[0] else ""
        print(f"  Проект {p+1}: {priority:.4f} ({priority*100:.2f}%){marker}")
    
    print(f"\n{'='*80}")
    print(f"ВИСНОВОК: Рекомендується ПРОЕКТ {best_project}")
    print(f"{'='*80}\n")
    
    print("Файли збережено в /mnt/user-data/outputs/:")
    print("  - lab5_results.txt (детальний звіт)")
    print("  - lab5_charts.png (графіки)")
    print("\n✓ Лабораторна робота №5 виконана успішно!")


if __name__ == "__main__":
    run_ahp_analysis()
