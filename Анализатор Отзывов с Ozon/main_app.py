import os
import threading
import traceback
from collections import Counter
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics.pairwise import cosine_similarity

# --- Локальные импорты вашего проекта ---
import neuralprocessing
from ozon_parser import process_and_save_reviews 
# --- КОНСТАНТЫ И НАСТРОЙКИ ---

# Путь к fine-tuned модели для суммаризации.
# ВАЖНО: Папка 'fine_tuned_rut5_large_third' должна лежать рядом с этим скриптом
# или нужно указать полный абсолютный путь.
RUT5_MODEL_PATH = os.path.join(os.path.dirname(__file__), "fine_tuned_rut5_large_third")

# Слова и фразы для исключения из анализа
SKIP_PHRASES = {"не указаны", "нет", "ошибка генерации", "хорошо", "хорошая", "хороший"}
ADDITIONAL_IGNORED_WORDS = {"всё", "супер", "нормально", "отлично", "класс"}
IGNORED_WORDS = list(SKIP_PHRASES.union(ADDITIONAL_IGNORED_WORDS))

# Доступные SBERT модели для кластеризации
SBERT_MODELS = {
    "paraphrase-multilingual-MiniLM-L12-v2": "Мультиязычная (MiniLM L12)",
    "paraphrase-multilingual-mpnet-base-v2": "Мультиязычная (MPNet Base)",
    "cointegrated/rubert-tiny2": "Русская (rubert-tiny2)",
}
DEFAULT_SBERT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Оптимальные параметры кластеризации (можно вынести в GUI для настройки)
OPTIMAL_DISTANCE_THRESHOLD = 0.4
OPTIMAL_LINKAGE = 'complete'
OPTIMAL_MIN_CLUSTER_SIZE = 2
OPTIMAL_LEMMATIZATION = True


class ReviewAnalyzerApp:
    """
    GUI-приложение для анализа отзывов на товары.
    Включает парсинг, обработку нейросетью и кластеризацию.
    """
    def __init__(self, root_window: tk.Tk):
        self.root = root_window
        self.root.title("Анализатор отзывов v1.3")
        self.root.geometry("650x400")

        # --- Состояние приложения ---
        self.all_results = None
        self.sbert_model = None
        self.selected_model_name = DEFAULT_SBERT_MODEL
        self.cached_embeddings = {}  # Кэш для эмбеддингов, чтобы не пересчитывать

        self._load_sbert_model(self.selected_model_name)
        self._create_widgets()

    def _load_sbert_model(self, model_name: str) -> bool:
        """Загружает SBERT модель для кластеризации."""
        try:
            print(f"Загрузка SBERT модели: {model_name}...")
            self.sbert_model = SentenceTransformer(model_name)
            self.selected_model_name = model_name
            self.cached_embeddings.clear()  # Сбрасываем кэш при смене модели
            print("SBERT модель успешно загружена.")
            return True
        except Exception as e:
            print(f"КРИТИЧЕСКАЯ ОШИБКА загрузки SBERT модели '{model_name}': {e}")
            traceback.print_exc()
            messagebox.showerror("Ошибка загрузки модели", f"Не удалось загрузить модель '{model_name}': {e}")
            return False

    def _create_widgets(self):
        """Создает и размещает все элементы интерфейса."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(expand=True, fill='both')

        # --- Блок ввода ссылки ---
        link_frame = ttk.Frame(main_frame)
        link_frame.pack(pady=5, fill='x')
        ttk.Label(link_frame, text="Ссылка на товар Ozon:").pack(side=tk.LEFT, padx=(0, 5))
        self.entry_link = ttk.Entry(link_frame, width=60)
        self.entry_link.pack(side=tk.LEFT, expand=True, fill='x')

        ttk.Button(main_frame, text="1. Спарсить и обработать отзывы", command=self.start_processing).pack(pady=5)

        # --- Блок выбора модели SBERT ---
        model_select_frame = ttk.Frame(main_frame)
        model_select_frame.pack(pady=10, fill='x')
        ttk.Label(model_select_frame, text="Модель SBERT для кластеризации:").pack(side=tk.LEFT, padx=(0, 5))

        self.model_var = tk.StringVar(value=self.selected_model_name)
        model_dropdown = ttk.Combobox(model_select_frame, textvariable=self.model_var, values=list(SBERT_MODELS.keys()), state="readonly", width=35)
        model_dropdown.pack(side=tk.LEFT, padx=5)
        ttk.Button(model_select_frame, text="2. Применить модель SBERT", command=self.change_sbert_model).pack(side=tk.LEFT, padx=5)

        # --- Блок кнопок с результатами ---
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="Показать все", command=self.show_all_reviews).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Топ-5", command=self.show_simple_top).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="3. Кластеры", command=self.show_clustered_top).pack(side=tk.LEFT, padx=2)

        self.progress_bar = ttk.Progressbar(main_frame, orient='horizontal', mode='determinate', length=300)
        self.progress_bar.pack(fill='x', pady=(10, 0))

    def update_progress(self, progress_value: int):
        """Обновляет значение прогресс-бара в главном потоке."""
        if self.root.winfo_exists():
            self.progress_bar["value"] = progress_value
            self.root.update_idletasks()

    def start_processing(self):
        """Запускает процесс парсинга и обработки отзывов в отдельном потоке."""
        link = self.entry_link.get().strip()
        if not link:
            messagebox.showerror("Ошибка", "Введите ссылку на товар.")
            return

        if not os.path.isdir(RUT5_MODEL_PATH):
            messagebox.showerror("Ошибка", f"Папка с моделью RuT5 не найдена по пути: {RUT5_MODEL_PATH}")
            return

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_name = f"reviews_{timestamp}"
            data_folder = os.path.join(os.path.dirname(__file__), "data")
            os.makedirs(data_folder, exist_ok=True)
            
            csv_filepath = os.path.join(data_folder, f'{dataset_name}.csv')

            # Запускаем парсинг
            process_and_save_reviews(link, dataset_name, output_folder=data_folder)

            if not os.path.exists(csv_filepath):
                messagebox.showerror("Ошибка", f"После парсинга не найден файл с данными: {csv_filepath}")
                return
            
            # Сбрасываем предыдущие результаты и запускаем обработку
            self.progress_bar["value"] = 0
            self.all_results = None
            self.cached_embeddings.clear()
            
            thread = threading.Thread(
                target=self._processing_thread_task,
                args=(csv_filepath, RUT5_MODEL_PATH)
            )
            thread.start()

        except Exception as e:
            print(f"КРИТИЧЕСКАЯ ОШИБКА в start_processing: {e}")
            traceback.print_exc()
            messagebox.showerror("Критическая ошибка", f"Произошла ошибка при запуске: {e}")

    def _processing_thread_task(self, csv_filepath: str, model_path: str):
        """Задача, выполняемая в отдельном потоке для обработки данных."""
        try:
            processing_results = neuralprocessing.main(
                csv_file_path=csv_filepath,
                model_path=model_path,
                progress_callback=self.update_progress
            )

            if not processing_results:
                self.root.after(0, messagebox.showinfo, "Информация", "Обработка завершена, но скрипт не вернул результатов.")
                return

            pluses = processing_results.get("Общие плюсы", {}).keys()
            minuses = processing_results.get("Общие минусы", {}).keys()
            
            self.all_results = {"Плюсы": list(pluses), "Минусы": list(minuses)}

            if self.root.winfo_exists():
                if self.all_results.get("Плюсы") or self.all_results.get("Минусы"):
                    message = "Обработка отзывов завершена. Данные для анализа загружены."
                else:
                    message = "Обработка завершена, но не удалось извлечь плюсы или минусы."
                self.root.after(0, messagebox.showinfo, "Успех", message)

        except Exception as e:
            print(f"КРИТИЧЕСКАЯ ОШИБКА в потоке _processing_thread_task: {e}")
            traceback.print_exc()
            if self.root.winfo_exists():
                self.root.after(0, messagebox.showerror, "Ошибка в потоке", f"Произошла ошибка: {e}")
            self.all_results = None

    def change_sbert_model(self):
        """Применяет новую SBERT модель, выбранную в выпадающем списке."""
        new_model_name = self.model_var.get()
        if new_model_name != self.selected_model_name:
            if self._load_sbert_model(new_model_name):
                display_name = SBERT_MODELS.get(new_model_name, new_model_name)
                messagebox.showinfo("Успех", f"Модель SBERT изменена на: {display_name}")
        else:
            messagebox.showinfo("Информация", "Эта модель уже выбрана.")

    # --- Методы отображения результатов ---

    def show_all_reviews(self):
        """Отображает окно со всеми извлеченными плюсами и минусами."""
        if not self._check_results_exist(): return
        
        ResultWindow(
            parent=self.root,
            title="Все извлеченные Плюсы и Минусы",
            content_data={
                "Плюсы": self.all_results["Плюсы"],
                "Минусы": self.all_results["Минусы"]
            }
        )
        
    def show_simple_top(self):
        """Отображает топ-5 плюсов и минусов по частоте упоминаний."""
        if not self._check_results_exist(): return
        
        pluses = [p for p in self.all_results.get("Плюсы", []) if p.lower() not in IGNORED_WORDS]
        minuses = [m for m in self.all_results.get("Минусы", []) if m.lower() not in IGNORED_WORDS]
        
        top_pluses = Counter(pluses).most_common(5)
        top_minuses = Counter(minuses).most_common(5)

        content = {
            "Топ-5 Плюсов": [f"'{phrase}' (упом. {count} раз)" for phrase, count in top_pluses],
            "Топ-5 Минусов": [f"'{phrase}' (упом. {count} раз)" for phrase, count in top_minuses]
        }
        
        ResultWindow(parent=self.root, title="Топ-5 (простая группировка)", content_data=content)

    def show_clustered_top(self):
        """Выполняет кластеризацию и показывает результат."""
        if not self._check_results_exist(): return

        clustered_data = self.get_clustered_results(
            distance_threshold=OPTIMAL_DISTANCE_THRESHOLD,
            linkage=OPTIMAL_LINKAGE,
            min_cluster_size=OPTIMAL_MIN_CLUSTER_SIZE,
            lemmatize=OPTIMAL_LEMMATIZATION
        )

        content = {}
        for category in ["Плюсы", "Минусы"]:
            representatives = clustered_data[category].get("представители", [])
            formatted_list = [f"'{phrase}' (группа из {count} фраз)" for phrase, count in representatives[:50]]
            if len(representatives) > 50:
                formatted_list.append(f"... и еще {len(representatives) - 50} групп ...")
            content[f"Кластеризованные {category}"] = formatted_list
            
        title = f"Кластеры (dist={OPTIMAL_DISTANCE_THRESHOLD}, link='{OPTIMAL_LINKAGE}', lem={OPTIMAL_LEMMATIZATION})"
        ResultWindow(parent=self.root, title=title, content_data=content)


    def get_clustered_results(self, distance_threshold: float, linkage: str, min_cluster_size: int, lemmatize: bool) -> dict:
        """
        Основной метод для выполнения кластеризации.
        Возвращает словарь с представителями кластеров для каждой категории.
        """
        clustered_output = {"Плюсы": {}, "Минусы": {}}
        
        for category in ["Плюсы", "Минусы"]:
            phrases = self.all_results.get(category, [])
            filtered_phrases = [p for p in phrases if isinstance(p, str) and p.strip() and p.lower() not in IGNORED_WORDS]
            
            if len(filtered_phrases) < 2:
                reps = [(p, 1) for p in filtered_phrases]
                clustered_output[category] = {"представители": reps, "метрики": {"сообщение": "Недостаточно данных."}}
                continue

            # Получение или создание эмбеддингов
            embeddings = self._get_or_create_embeddings(category, filtered_phrases, lemmatize)
            
            # Кластеризация
            cluster_labels = self._perform_clustering(embeddings, distance_threshold, linkage)
            
            # Извлечение представителей кластеров
            representatives = self._extract_cluster_representatives(filtered_phrases, cluster_labels, min_cluster_size)
            
            clustered_output[category] = {"представители": representatives, "метрики": {}} # Метрики можно добавить сюда
            
        return clustered_output

    def _get_or_create_embeddings(self, category: str, phrases: list, lemmatize: bool) -> np.ndarray:
        """Возвращает эмбеддинги из кэша или создает и кэширует их."""
        cache_key = (category, lemmatize, tuple(phrases))
        if cache_key in self.cached_embeddings:
            return self.cached_embeddings[cache_key]

        phrases_to_encode = phrases
        if lemmatize:
            try:
                import pymorphy2
                morph = pymorphy2.MorphAnalyzer()
                phrases_to_encode = [" ".join([morph.parse(word)[0].normal_form for word in p.split()]) for p in phrases]
            except ImportError:
                print("ПРЕДУПРЕЖДЕНИЕ: pymorphy2 не установлен. Лемматизация не будет выполнена.")
        
        embeddings = self.sbert_model.encode(phrases_to_encode, show_progress_bar=False)
        self.cached_embeddings[cache_key] = embeddings
        return embeddings

    def _perform_clustering(self, embeddings: np.ndarray, distance_threshold: float, linkage: str) -> np.ndarray:
        """Выполняет агломеративную кластеризацию."""
        distance_matrix = 1 - cosine_similarity(embeddings)
        np.fill_diagonal(distance_matrix, 0)
        distance_matrix[distance_matrix < 0] = 0

        model = AgglomerativeClustering(
            n_clusters=None,
            metric='precomputed',
            linkage=linkage,
            distance_threshold=distance_threshold
        )
        return model.fit_predict(distance_matrix)

    def _extract_cluster_representatives(self, original_phrases: list, cluster_labels: np.ndarray, min_cluster_size: int) -> list:
        """
        Группирует фразы по кластерам и выбирает кратчайшую фразу как представителя.
        """
        cluster_groups = {}
        for i, label in enumerate(cluster_labels):
            if label not in cluster_groups:
                cluster_groups[label] = []
            cluster_groups[label].append(original_phrases[i])
            
        representatives = []
        # Сортируем кластеры по размеру, чтобы самые большие были первыми
        sorted_clusters = sorted(cluster_groups.values(), key=len, reverse=True)

        for cluster in sorted_clusters:
            if len(cluster) >= min_cluster_size:
                # Представитель - самая короткая фраза в кластере
                representative = min(cluster, key=len)
                representatives.append((representative, len(cluster)))
            else:
                # Если кластер слишком мал, каждая фраза сама по себе
                representatives.extend([(phrase, 1) for phrase in cluster])
        return representatives

    def _check_results_exist(self) -> bool:
        """Проверяет, есть ли данные для отображения."""
        if not self.all_results or not (self.all_results.get("Плюсы") or self.all_results.get("Минусы")):
            messagebox.showinfo("Нет данных", "Сначала необходимо обработать отзывы.")
            return False
        return True


class ResultWindow(tk.Toplevel):
    """Универсальное окно для отображения результатов."""
    def __init__(self, parent, title: str, content_data: dict):
        super().__init__(parent)
        self.title(title)
        self.geometry("800x600")

        text_widget = tk.Text(self, wrap=tk.WORD, font=("Arial", 10), padx=10, pady=10)
        text_widget.pack(expand=True, fill='both')

        # Настройка стилей
        text_widget.tag_configure('header', font=('Arial', 12, 'bold'), spacing3=10)

        for header, lines in content_data.items():
            text_widget.insert(tk.END, f"=== {header.upper()} ===\n", 'header')
            if lines:
                for line in lines:
                    text_widget.insert(tk.END, f"- {line}\n")
            else:
                text_widget.insert(tk.END, "- Нет данных для отображения\n")
            text_widget.insert(tk.END, "\n")
            
        text_widget.config(state=tk.DISABLED) # Запрещаем редактирование


if __name__ == '__main__':
    root = tk.Tk()
    app = ReviewAnalyzerApp(root)
    root.mainloop()