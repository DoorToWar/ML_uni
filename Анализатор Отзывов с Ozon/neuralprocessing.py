import gc
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from typing import List, Dict, Optional, Callable, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- КОНСТАНТЫ И НАСТРОЙКИ ---

# Параметры для генерации текста моделью T5
GENERATION_CONFIG = {
    "max_new_tokens": 80,
    "min_length": 15,
    "num_return_sequences": 1,
    "temperature": 0.7,
    "do_sample": True,
    "repetition_penalty": 2.5,
    "no_repeat_ngram_size": 3,
    "top_k": 50,
    "top_p": 0.95,
    "num_beams": 5,
    "early_stopping": True,
    "length_penalty": 1.0
}

# Слова, которые следует игнорировать при анализе и группировке
SKIP_PHRASES = {"не указаны", "нет", "ошибка генерации", "хорошо", "хорошая", "хороший"}
DEFAULT_REVIEW_COLUMN = "Полный комментарий"


class ReviewProcessor:
    """
    Класс для обработки текстовых отзывов:
    1. Суммаризация каждого отзыва с выделением плюсов и минусов.
    2. Группировка и подсчет схожих плюсов и минусов.
    3. Сбор и вывод описательной статистики по результатам.
    """

    def __init__(self, model_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Используемое устройство для обработки: {self.device}")

        print(f"Загрузка модели из: {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        print("Модель успешно загружена.")

    def process_reviews_from_file(self,
                                  csv_file_path: str,
                                  review_column: str = DEFAULT_REVIEW_COLUMN,
                                  num_threads: int = 4,
                                  similarity_threshold: float = 0.75,
                                  progress_callback: Optional[Callable[[int], None]] = None
                                  ) -> Optional[Dict[str, Dict[str, int]]]:
        """
        Основной метод для обработки отзывов из CSV файла.
        Возвращает словарь, совместимый с GUI.
        """
        try:
            reviews = self._load_reviews_from_csv(csv_file_path, review_column)
            if not reviews:
                return {"Общие плюсы": {}, "Общие минусы": {}}
        except (FileNotFoundError, ValueError) as e:
            print(f"Ошибка при чтении или валидации CSV файла: {e}")
            if progress_callback: progress_callback(100)
            return None

        print(f"Начало обработки {len(reviews)} отзывов...")
        
        structured_summaries = self._run_parallel_summarization(reviews, num_threads, progress_callback)
        analysis_results = self._analyze_summaries(structured_summaries, similarity_threshold)

        print("Обработка завершена.")
        return analysis_results

    def _load_reviews_from_csv(self, file_path: str, column_name: str) -> List[str]:
        """Загружает и валидирует отзывы из CSV файла."""
        df = pd.read_csv(file_path)
        if column_name not in df.columns:
            raise ValueError(f"Столбец '{column_name}' не найден в файле.")
        
        reviews = [str(review) for review in df[column_name] if pd.notna(review)]
        if not reviews:
            print("В файле не найдено ни одного отзыва для обработки.")
        return reviews

    def _run_parallel_summarization(self,
                                    reviews: List[str],
                                    num_threads: int,
                                    progress_callback: Optional[Callable[[int], None]]
                                    ) -> List[Dict[str, str]]:
        """Выполняет суммаризацию отзывов в несколько потоков."""
        summaries = []
        tasks = [(review, i, len(reviews)) for i, review in enumerate(reviews)]

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_task = {executor.submit(self._summarize_single_review, task, progress_callback): task for task in tasks}
            for future in as_completed(future_to_task):
                try:
                    summary = future.result()
                    summaries.append(summary)
                except Exception as e:
                    failed_task_index = future_to_task[future][1]
                    print(f"Ошибка в потоке при обработке отзыва #{failed_task_index}: {e}")
                    summaries.append({"Плюсы": "Ошибка потока", "Минусы": "Ошибка потока", "Полное саммари": f"Ошибка: {e}"})
        return summaries

    def _summarize_single_review(self,
                                 task_data: Tuple[str, int, int],
                                 progress_callback: Optional[Callable[[int], None]]
                                 ) -> Dict[str, str]:
        """Генерирует краткое содержание для одного отзыва."""
        review_text, review_index, total_reviews = task_data
        try:
            prompt = "summarize: " + review_text
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model.generate(**inputs, **GENERATION_CONFIG)
            raw_summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            pluses, minuses = self._parse_summary(raw_summary)

            return {"Плюсы": pluses, "Минусы": minuses, "Полное саммари": raw_summary}

        except Exception as e:
            print(f"Ошибка при генерации саммари для отзыва #{review_index}: {e}")
            return {"Плюсы": "Ошибка генерации", "Минусы": "Ошибка генерации", "Полное саммари": f"Ошибка: {e}"}
        finally:
            if progress_callback:
                progress = int(((review_index + 1) / total_reviews) * 100)
                progress_callback(progress)

    @staticmethod
    def _parse_summary(summary_text: str) -> Tuple[str, str]:
        """Извлекает плюсы и минусы из сгенерированного текста с помощью regex."""
        cleaned = re.sub(r'<extra_id_\d+>|\s+', ' ', summary_text).strip()
        pluses, minuses = "Не указаны", "Не указаны"

        # Ищет "Плюсы" и текст до "Минусы" (или до конца строки)
        match_pluses = re.search(r'(Плюсы|Достоинства)\s*:\s*(.*?)(?=(Минусы|Недостатки)\s*:|$)', cleaned, re.IGNORECASE | re.DOTALL)
        
        if match_pluses:
            pluses = match_pluses.group(2).strip()
        
        # Ищет "Минусы" и текст после них
        match_minuses = re.search(r'(Минусы|Недостатки)\s*:\s*(.*)', cleaned, re.IGNORECASE | re.DOTALL)

        if match_minuses:
            minuses = match_minuses.group(2).strip()
        elif not match_pluses:  # Если не нашли ни "Плюсы", ни "Минусы", считаем все плюсами
            pluses = re.sub(r'^(summarize|обзор|саммари)\s*:?\s*', '', cleaned, flags=re.IGNORECASE).strip()

        # Финальная очистка и проверка на пустые строки
        if not pluses or pluses.lower() == "нет": pluses = "Не указаны"
        if not minuses or minuses.lower() == "нет": minuses = "Не указаны"

        return pluses, minuses

    def _analyze_summaries(self, summaries: List[Dict[str, str]], similarity_threshold: float) -> Dict[str, Dict[str, int]]:
        """Анализирует все саммари, группирует, подсчитывает и выводит статистику."""
        
        def is_valid_aspect(text: str) -> bool:
            """Проверяет, является ли фраза осмысленной, а не 'заглушкой'."""
            return text.lower().strip() not in SKIP_PHRASES

        # Собираем списки валидных фраз
        all_pluses = [s['Плюсы'] for s in summaries if is_valid_aspect(s.get('Плюсы', ''))]
        all_minuses = [s['Минусы'] for s in summaries if is_valid_aspect(s.get('Минусы', ''))]

        # Группировка
        plus_counts = self._group_phrases_by_similarity(all_pluses, similarity_threshold)
        minus_counts = self._group_phrases_by_similarity(all_minuses, similarity_threshold)
        
        # Расчет и вывод статистики
        self._calculate_and_print_stats(summaries, all_pluses, all_minuses)

        return {
            "Общие плюсы": dict(plus_counts),
            "Общие минусы": dict(minus_counts),
        }

    @staticmethod
    def _calculate_string_similarity(a: str, b: str) -> float:
        """Вычисляет схожесть двух строк."""
        return SequenceMatcher(None, a, b).ratio()

    def _group_phrases_by_similarity(self, phrases: List[str], threshold: float) -> List[Tuple[str, int]]:
        """Группирует список фраз по лексической схожести."""
        if not phrases:
            return []
            
        # Сортировка по длине для выбора более длинных и информативных представителей
        sorted_phrases = sorted(phrases, key=len, reverse=True)
        groups = []

        for phrase in sorted_phrases:
            lower_phrase = phrase.lower()
            found_group = False
            for group in groups:
                representative = group[0].lower()
                if self._calculate_string_similarity(lower_phrase, representative) >= threshold:
                    group.append(phrase)
                    found_group = True
                    break
            if not found_group:
                groups.append([phrase])
        
        # Сортируем группы по убыванию их размера
        counted_groups = sorted(
            [(group[0], len(group)) for group in groups],
            key=lambda item: item[1],
            reverse=True
        )
        return counted_groups

    @staticmethod
    def _calculate_and_print_stats(summaries: List[Dict], valid_pluses: List, valid_minuses: List):
        """Считает и выводит в консоль описательную статистику по генерации."""
        total = len(summaries)
        if total == 0:
            return

        pluses_found = len(valid_pluses)
        minuses_found = len(valid_minuses)
        both_found = sum(1 for s in summaries if s['Плюсы'].lower() not in SKIP_PHRASES and s['Минусы'].lower() not in SKIP_PHRASES)

        pluses_lengths = [len(p.split()) for p in valid_pluses]
        minuses_lengths = [len(m.split()) for m in valid_minuses]
        full_summary_lengths = [len(s['Полное саммари'].split()) for s in summaries if s.get('Полное саммари')]

        stats = {
            "Всего обработано отзывов": total,
            "Извлечено валидных плюсов (%)": f"{pluses_found} ({pluses_found / total:.2%})",
            "Извлечено валидных минусов (%)": f"{minuses_found} ({minuses_found / total:.2%})",
            "Извлечено и плюсов, и минусов (%)": f"{both_found} ({both_found / total:.2%})",
            "Средняя/медианная длина плюсов (слов)": f"{np.mean(pluses_lengths):.1f} / {np.median(pluses_lengths):.1f}" if pluses_lengths else "0.0 / 0.0",
            "Средняя/медианная длина минусов (слов)": f"{np.mean(minuses_lengths):.1f} / {np.median(minuses_lengths):.1f}" if minuses_lengths else "0.0 / 0.0",
            "Средняя длина полного саммари (слов)": f"{np.mean(full_summary_lengths):.1f}" if full_summary_lengths else "0.0",
        }
        
        print("\n--- Описательная статистика по генерации саммари ---")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        print("----------------------------------------------------")


def main(csv_file_path, model_path, progress_callback=None, **kwargs):
    """
    Основная функция-обертка для запуска из других модулей (например, GUI).
    """
    processor = None
    try:
        processor = ReviewProcessor(model_path=model_path)
        results = processor.process_reviews_from_file(
            csv_file_path=csv_file_path,
            progress_callback=progress_callback,
            **kwargs
        )
        return results
    except Exception as e:
        print(f"Критическая ошибка в процессе выполнения: {e}")
        return None
    finally:
        # Освобождение памяти GPU, даже если произошла ошибка
        if processor and processor.device == 'cuda':
            del processor.model
            del processor.tokenizer
            torch.cuda.empty_cache()
            print("Память GPU очищена.")
        gc.collect()


if __name__ == '__main__':
    # --- Пример локального запуска для тестирования ---
    TEST_CSV_PATH = 'data/sample.csv'
    MODEL_PATH = 'fine_tuned_rut5_large_third_v3' # Убедитесь, что путь верный

    def simple_progress_bar(progress: int):
        bar = '█' * (progress // 4) + '░' * (25 - progress // 4)
        print(f'\rОбработка: [{bar}] {progress}%', end='')

    if not os.path.exists(TEST_CSV_PATH):
        print(f"Создание тестового файла {TEST_CSV_PATH}...")
        os.makedirs('data', exist_ok=True)
        sample_data = {DEFAULT_REVIEW_COLUMN: [
            "Телефон просто супер. Камера отличная, батарея держит долго. Очень доволен покупкой.",
            "В целом неплохо. Плюсы: яркий экран, быстрый процессор. Минусы: сильно греется при играх.",
            "Не рекомендую. Сломался через неделю использования. Вернул по гарантии.",
            "Хороший аппарат за свои деньги. Доставка быстрая, упаковка целая.",
            "Качество сборки оставляет желать лучшего, корпус скрипит. Батареи на день не хватает."
        ]}
        pd.DataFrame(sample_data).to_csv(TEST_CSV_PATH, index=False)
        
    print("--- Запуск тестовой обработки ---")
    final_results = main(
        csv_file_path=TEST_CSV_PATH,
        model_path=MODEL_PATH,
        progress_callback=simple_progress_bar,
        num_threads=2 # Для локальных тестов
    )

    if final_results:
        print("\n\n--- Результаты анализа для GUI ---")
        print("\nОбщие плюсы:")
        for phrase, count in final_results.get("Общие плюсы", {}).items():
            print(f"- '{phrase}' (сгруппировано {count} раз)")

        print("\nОбщие минусы:")
        for phrase, count in final_results.get("Общие минусы", {}).items():
            print(f"- '{phrase}' (сгруппировано {count} раз)")