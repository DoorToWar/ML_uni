import gc
import os
import re
from typing import Dict, List, Optional

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    EarlyStoppingCallback,
)
import nltk

# --- БЛОК НАСТРОЙКИ РЕСУРСОВ NLTK ---

def download_nltk_resource(resource_name: str, resource_path: str):
    """Проверяет наличие и при необходимости загружает ресурс NLTK."""
    try:
        nltk.data.find(resource_path)
        print(f"[NLTK] Ресурс '{resource_name}' найден.")
    except LookupError:
        print(f"[NLTK] Ресурс '{resource_name}' не найден. Загрузка...")
        try:
            nltk.download(resource_name, quiet=False)
            print(f"[NLTK] Ресурс '{resource_name}' успешно загружен.")
        except Exception as e:
            print(f"[NLTK] Ошибка загрузки '{resource_name}': {e}")
            print(f"  Пожалуйста, попробуйте загрузить вручную: import nltk; nltk.download('{resource_name}')")

print("Проверка необходимых ресурсов NLTK для метрик...")
download_nltk_resource('punkt', 'tokenizers/punkt')
download_nltk_resource('wordnet', 'corpora/wordnet')
download_nltk_resource('omw-1.4', 'corpora/omw-1.4')
print("Проверка NLTK завершена.")


# --- БЛОК ПОДГОТОВКИ ДАННЫХ ---

def clean_text(text: str) -> str:
    """Базовая очистка текста от лишних символов и пробелов."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[—–«»→]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_prepare_data(csv_file_path: str, sample_size: Optional[int] = None, task_prefix: str = "summarize: ") -> Optional[pd.DataFrame]:
    """Загружает, очищает и фильтрует данные из CSV файла."""
    print(f"Загрузка данных из {csv_file_path}...")
    try:
        df = pd.read_csv(csv_file_path)
        if "text" not in df.columns or "summary" not in df.columns:
            raise ValueError("CSV файл должен содержать колонки 'text' и 'summary'.")

        initial_count = len(df)
        print(f"Исходное количество записей: {initial_count}")

        df['text'] = df['text'].apply(lambda x: task_prefix + clean_text(str(x)))
        df['summary'] = df['summary'].apply(clean_text)

        # Фильтрация по наличию и длине
        df.dropna(subset=["text", "summary"], inplace=True)
        df = df[(df["text"].str.len() > 20) & (df["summary"].str.len() > 15)]
        print(f"Записей после удаления пустых и коротких строк: {len(df)}")
        
        # Фильтрация по формату "Плюсы: ... Минусы: ..."
        pattern = r'Плюсы:\s*.+\s*Минусы:\s*.+'
        df = df[df["summary"].str.match(pattern, case=False, flags=re.DOTALL)]
        print(f"Записей после фильтрации по формату 'Плюсы/Минусы': {len(df)}")

        df.drop_duplicates(subset=["text"], inplace=True)
        print(f"Записей после удаления дубликатов по 'text': {len(df)}")

        if not df.empty and sample_size and 0 < sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            print(f"Итоговый размер выборки ограничен до {len(df)} записей.")

        if df.empty:
            print("После фильтрации не осталось данных для обучения.")
            return None

        return df.reset_index(drop=True)

    except (FileNotFoundError, ValueError) as e:
        print(f"Ошибка при загрузке или подготовке данных: {e}")
        return None


# --- БЛОК ОБУЧЕНИЯ МОДЕЛИ ---

class CustomLoggingCallback(TrainerCallback):
    """Коллбэк для вывода логов обучения в компактном виде."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and logs:
            log_str = f"Epoch: {logs.get('epoch', 0):.2f} | Step: {state.global_step}"
            if 'loss' in logs:
                log_str += f" | Loss: {logs['loss']:.4f}"
            if 'learning_rate' in logs:
                log_str += f" | LR: {logs['learning_rate']:.2e}"
            for k, v in logs.items():
                if k.startswith("eval_"):
                    log_str += f" | {k[5:].capitalize()}: {v:.4f}"
            print(log_str)


class ModelTrainer:
    """Класс, инкапсулирующий весь процесс дообучения модели."""

    def __init__(self, model_name: str, output_dir: str):
        self.model_name = model_name
        self.output_dir = output_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.metrics = evaluate.combine(["rouge", "bleu", "meteor", "bertscore"])

        self.model.config.use_cache = False
        if torch.cuda.is_available():
            self.model.to("cuda")
            print(f"Модель '{model_name}' перемещена на GPU.")

    def _preprocess_function(self, examples: Dict, max_input_len: int, max_target_len: int) -> Dict:
        """Токенизирует входные и целевые тексты."""
        model_inputs = self.tokenizer(
            examples["text"], max_length=max_input_len, truncation=True, padding="max_length"
        )
        labels = self.tokenizer(
            text_target=examples["summary"], max_length=max_target_len, truncation=True, padding="max_length"
        )
        # Заменяем pad_token_id в метках на -100, чтобы они игнорировались при расчете loss
        model_inputs["labels"] = [
            [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
        return model_inputs

    def _compute_metrics(self, eval_preds):
        """Расчет метрик качества на основе предсказаний модели."""
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        
        # Декодируем предсказания
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        
        # Заменяем -100 на pad_token_id для декодирования реальных меток
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Вычисляем метрики
        result = self.metrics.compute(predictions=decoded_preds, references=decoded_labels)
        
        # Добавляем длину предсказаний для информации
        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in result.items()}

    def train(self, df: pd.DataFrame, training_args: Seq2SeqTrainingArguments):
        """Основной метод, запускающий процесс обучения."""
        train_df, eval_df = train_test_split(df, test_size=0.15, random_state=42)
        train_dataset = Dataset.from_pandas(train_df)
        eval_dataset = Dataset.from_pandas(eval_df)
        print(f"Данные разделены на обучающую ({len(train_dataset)}) и валидационную ({len(eval_dataset)}) выборки.")
        
        # Динамический расчет длин на основе 95-го перцентиля
        max_input_len = int(np.percentile([len(self.tokenizer.encode(t)) for t in train_df["text"]], 95))
        max_target_len = int(np.percentile([len(self.tokenizer.encode(s)) for s in train_df["summary"]], 95))
        max_input_len = min(max_input_len, self.tokenizer.model_max_length)
        max_target_len = min(max_target_len, 256) # Ограничиваем сверху
        print(f"Рассчитанная макс. длина: вход={max_input_len}, цель={max_target_len}")

        # Применяем токенизацию к датасетам
        tokenized_train = train_dataset.map(
            lambda x: self._preprocess_function(x, max_input_len, max_target_len),
            batched=True, remove_columns=train_df.columns.tolist()
        )
        tokenized_eval = eval_dataset.map(
            lambda x: self._preprocess_function(x, max_input_len, max_target_len),
            batched=True, remove_columns=eval_df.columns.tolist()
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorForSeq2Seq(self.tokenizer, model=self.model),
            compute_metrics=self._compute_metrics,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001),
                CustomLoggingCallback()
            ]
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        print("\n--- Начало процесса обучения ---")
        trainer.train()
        print("\n--- Обучение завершено ---")
        
        print("\n--- Финальная оценка лучшей модели ---")
        eval_results = trainer.evaluate()
        print(eval_results)
        
        print(f"\nСохранение модели и токенизатора в '{self.output_dir}'...")
        trainer.save_model(self.output_dir)
        print("Модель успешно сохранена.")


def main():
    """Главная функция для запуска всего пайплайна обучения."""
    # --- НАСТРОЙКИ ОБУЧЕНИЯ ---
    CSV_FILE_PATH = "data/full_reviews.csv"       # Путь к вашему датасету
    BASE_MODEL = "ai-forever/ruT5-large"          # Модель для дообучения
    OUTPUT_DIR = "fine_tuned_rut5_large_final"    # Куда сохранить результат
    SAMPLE_SIZE = 2000                            # None для использования всего датасета, или число для выборки
    NUM_EPOCHS = 3
    
    # Создаем папку для вывода, если ее нет
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Загрузка и подготовка данных
    df = load_and_prepare_data(CSV_FILE_PATH, sample_size=SAMPLE_SIZE)
    if df is None:
        return

    # Настройки для тренера
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        # Стратегии
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        # Гиперпараметры
        learning_rate=3e-5,
        per_device_train_batch_size=2, # Уменьшите, если не хватает VRAM
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8, # Эффективный батч = 2 * 8 = 16
        weight_decay=0.01,
        num_train_epochs=NUM_EPOCHS,
        optim="adafactor",
        # Оптимизация памяти
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        # Оценка
        predict_with_generate=True,
        report_to="tensorboard",
        load_best_model_at_end=True,
        metric_for_best_model="meteor",
        greater_is_better=True,
        generation_max_length=256,
        generation_num_beams=4,
    )
    
    trainer = ModelTrainer(model_name=BASE_MODEL, output_dir=OUTPUT_DIR)
    trainer.train(df, training_args)

if __name__ == "__main__":
    main()