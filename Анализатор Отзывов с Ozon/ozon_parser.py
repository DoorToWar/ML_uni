import logging
import os
import random
import re
import time
from typing import Dict, List, Optional, Union

import emoji
import pandas as pd
import undetected_chromedriver as uc
from selenium.common.exceptions import (
    ElementClickInterceptedException,
    NoSuchElementException,
    TimeoutException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# --- НАСТРОЙКА ЛОГИРОВАНИЯ ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler("ozon_parser.log", mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class OzonReviewParser:
    """
    Класс для парсинга отзывов с товарных страниц Ozon.
    Управляет браузером, обходит страницы и извлекает структурированные данные.
    """
    # Стабильные селекторы для элементов страницы
    REVIEW_ELEMENT_SELECTOR = (By.XPATH, "//div[@data-widget='webListReviews']//*[@data-review-uuid]")
    READ_MORE_BUTTON_SELECTOR = (By.XPATH, ".//span[contains(text(), 'Читать полностью')]")
    NEXT_PAGE_BUTTON_SELECTOR = (By.XPATH, "//a[contains(., 'Дальше')] | //button[contains(., 'Дальше')]")
    
    # Ключевые слова для структурированных отзывов
    STRUCTURE_MARKERS = r"(Достоинства|Недостатки|Комментарий|Плюсы|Минусы)"
    
    def __init__(self, headless: bool = True):
        """
        Инициализирует парсер.

        Args:
            headless (bool): Запускать ли браузер в фоновом ("невидимом") режиме.
        """
        self.options = uc.ChromeOptions()
        if headless:
            self.options.add_argument('--headless=new')
        self.options.add_argument('--disable-gpu')
        self.options.add_argument('--no-sandbox')
        self.driver = None

    def __enter__(self):
        """Контекстный менеджер для инициализации драйвера."""
        logging.info("Запуск браузера...")
        self.driver = uc.Chrome(options=self.options)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Контекстный менеджер для корректного закрытия драйвера."""
        if self.driver:
            logging.info("Закрытие браузера...")
            self.driver.quit()

    @staticmethod
    def _clean_text(text: str) -> str:
        """Применяет базовую очистку к тексту: убирает эмодзи, лишние пробелы и т.д."""
        if not isinstance(text, str):
            return ""
        text = emoji.replace_emoji(text, replace='')
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _expand_review(self, review_element: WebElement, timeout: int = 2):
        """Раскрывает полный текст отзыва, если есть кнопка 'Читать полностью'."""
        try:
            read_more_button = review_element.find_element(*self.READ_MORE_BUTTON_SELECTOR)
            self.driver.execute_script("arguments[0].click();", read_more_button)
            # Ждем, пока кнопка исчезнет (признак раскрытия отзыва)
            WebDriverWait(review_element, timeout).until(
                EC.invisibility_of_element_located(self.READ_MORE_BUTTON_SELECTOR)
            )
        except (NoSuchElementException, TimeoutException, ElementClickInterceptedException):
            # Кнопки нет, или она не исчезла - не страшно, продолжаем
            pass

    def _parse_single_review(self, review_element: WebElement) -> Dict[str, str]:
        """Извлекает и структурирует данные из одного элемента отзыва."""
        self._expand_review(review_element)
        full_text = review_element.text

        # Удаляем информацию о "полезности", которая всегда в конце
        helpful_pattern = r"вам помог этот отзыв.*"
        cleaned_full_text = re.sub(helpful_pattern, "", full_text, flags=re.DOTALL | re.IGNORECASE).strip()

        pluses, minuses, comment = "", "", ""
        
        # Ищем структурированные части
        parts = re.findall(f"{self.STRUCTURE_MARKERS}\\s*(.*?)(?=(?:\\s*{self.STRUCTURE_MARKERS})|\\s*$)",
                           cleaned_full_text, flags=re.DOTALL | re.IGNORECASE)

        if parts:
            # Отзыв структурирован
            for marker, content in parts:
                cleaned_content = self._clean_text(content)
                marker_lower = marker.lower()
                if "достоинства" in marker_lower or "плюсы" in marker_lower:
                    pluses = cleaned_content
                elif "недостатки" in marker_lower or "минусы" in marker_lower:
                    minuses = cleaned_content
                elif "комментарий" in marker_lower:
                    comment = cleaned_content
        else:
            # Отзыв неструктурирован, весь текст - комментарий
            # Удаляем имя пользователя, дату и прочую "шапку"
            # Обычно это первые 1-2 строки
            lines = cleaned_full_text.split('\n')
            if len(lines) > 2:
                comment_text = "\n".join(lines[2:])
                comment = self._clean_text(comment_text)
            else:
                comment = self._clean_text(cleaned_full_text)

        return {
            "Достоинства": pluses,
            "Недостатки": minuses,
            "Комментарий": comment,
            "Полный комментарий": self._clean_text(cleaned_full_text),
        }

    def scrape_product_reviews(self, product_url: str, max_pages: int = 10) -> pd.DataFrame:
        """
        Собирает все отзывы для одного товара, переходя по страницам.

        Args:
            product_url (str): Ссылка на страницу товара.
            max_pages (int): Максимальное количество страниц для парсинга.

        Returns:
            pd.DataFrame с собранными отзывами.
        """
        if 'reviews' not in product_url:
            product_url = product_url.split('?')[0] + 'reviews/'

        self.driver.get(product_url)
        all_reviews_data = []

        for page_num in range(1, max_pages + 1):
            logging.info(f"Обработка страницы {page_num} для: {product_url[:50]}...")
            try:
                review_elements = WebDriverWait(self.driver, 15).until(
                    EC.presence_of_all_elements_located(self.REVIEW_ELEMENT_SELECTOR)
                )
                
                for element in review_elements:
                    all_reviews_data.append(self._parse_single_review(element))
                
                # Поиск и переход на следующую страницу
                next_button = self.driver.find_element(*self.NEXT_PAGE_BUTTON_SELECTOR)
                self.driver.execute_script("arguments[0].scrollIntoView(true);", next_button)
                time.sleep(0.5) # Даем время на прокрутку
                next_button.click()
                time.sleep(random.uniform(2, 4)) # Пауза для загрузки новой страницы
                
            except TimeoutException:
                logging.warning("Отзывы на странице не найдены или не загрузились.")
                break
            except NoSuchElementException:
                logging.info("Достигнута последняя страница с отзывами.")
                break
            except Exception as e:
                logging.error(f"Произошла ошибка на странице {page_num}: {e}")
                break
        
        return pd.DataFrame(all_reviews_data)


def process_and_save_reviews(links: Union[str, List[str]], dataset_name: str, output_folder: str = "data"):
    """
    Основная функция-обертка для запуска парсинга и сохранения результатов.

    Args:
        links (Union[str, List[str]]): Одна ссылка или список ссылок на товары.
        dataset_name (str): Базовое имя для сохраняемых CSV файлов.
        output_folder (str): Папка для сохранения результатов.
    """
    if not isinstance(links, list):
        links = [links]

    os.makedirs(output_folder, exist_ok=True)

    with OzonReviewParser(headless=True) as parser:
        for i, link in enumerate(links):
            logging.info(f"--- Начало работы со ссылкой {i+1}/{len(links)} ---")
            df_reviews = parser.scrape_product_reviews(link, max_pages=20)
            
            if not df_reviews.empty:
                filename = f"{dataset_name}_{i+1}.csv" if len(links) > 1 else f"{dataset_name}.csv"
                filepath = os.path.join(output_folder, filename)
                df_reviews.to_csv(filepath, index=False, encoding='utf-8-sig')
                logging.info(f"Сохранено {len(df_reviews)} отзывов в файл: {filepath}")
            else:
                logging.warning(f"Не удалось собрать отзывы для ссылки: {link}")


if __name__ == '__main__':
    # --- Пример использования ---
    monitor_links = [
        'https://www.ozon.ru/product/samsung-27-monitor-essential-s3-ls27c330gaixci-chernyy-1694843847/',
        'https://www.ozon.ru/product/samsung-25-monitor-odyssey-g4-ls25bg400eixci-chernyy-1646295320/',
        'https://www.ozon.ru/product/digma-23-8-monitor-progress-24p404f-chernyy-matovyy-1379391214/',
    ]
    
    # Ваш GUI будет вызывать эту функцию
    # Обратите внимание: она теперь принимает параметр output_folder
    # processReviewsByLink(link_input, dataset_name, output_folder=data_folder)
    # нужно будет заменить на
    # process_and_save_reviews(link_input, dataset_name, output_folder=data_folder)
    
    process_and_save_reviews(monitor_links, "monitors_reviews")
    
    # Пример для одной ссылки
    # process_and_save_reviews('https://www.ozon.ru/product/igrovaya-konsol-playstation-5-slim-blu-ray-1316509567/', 'ps5_reviews')