import ast
import json
import os
import re
import sqlite3
from collections import defaultdict, Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Dict, Any

import ollama
import yaml
from tqdm import tqdm

import pandas as pd


#from llama_cpp import Llama

class MetricType(Enum):
    COUNT_BY_TAG = "count_by_tag"
    TOP_N_TAGS = "top_n_tags"
    TAG_TRENDS = "tag_trends"
    COMPARISON = "comparison"
    SUMMARY_STATS = "summary_stats"
    COUNT_BY_SEMANTIC = "count_by_semantic"
    SEMANTIC_TRENDS = "semantic_trends"

@dataclass
class AnalysisPlan:
    time_period: Dict[str, Any]
    target_tags: List[str]
    metrics: List[MetricType]
    keywords: List[str]
    keyword_metrics: List[MetricType]
    semantic_queries: List[str] = None   
    semantic_metrics: List[MetricType] = None
    grouping: str = "month"
    comparison_tags: List[str] = None
    additional_filters: Dict = None

    def to_dict(self):
        return {
            'time_period': self.time_period,
            'target_tags': self.target_tags,
            'metrics': [m.value for m in self.metrics],
            'keywords': self.keywords,
            'keyword_metrics': [m.value for m in self.keyword_metrics],
            'semantic_queries': self.semantic_queries,
            'semantic_metrics': [m.value for m in self.semantic_metrics] if self.semantic_metrics else [],
            'grouping': self.grouping,
            'comparison_tags': self.comparison_tags,
            'filters': self.additional_filters or {}
        }

class DriveDataLoader:
    def __init__(self, json_directory: str, drive_path: str = None):
        self.csv_dir = json_directory
        self.drive_path = drive_path
        self.calls_cache = None
        self.conn = None
        self.timeout=600
        print(f"data loader timeout {self.timeout}")


    def load_all_calls(self, limit: int = None) -> List[Dict]:
        if self.calls_cache is not None:
            return self.calls_cache[:limit] if limit else self.calls_cache

        all_calls = []

        if not os.path.exists(self.csv_dir):
            print(f"Директория не найдена: {self.csv_dir}")
            if self.drive_path:
                print(f"Ожидаемый путь: {self.csv_dir}")
            return []

        try:
            csv_files = [f for f in os.listdir(self.csv_dir) if f.endswith('.csv')]
        except Exception as e:
            print(f"Ошибка чтения директории: {e}")
            return []

        if not csv_files:
            print(f"В директории {self.csv_dir} нет CSV файлов")
            return []

        csv_file = csv_files[0]
        filepath = os.path.join(self.csv_dir, csv_file)

        print(f" Читаю данные из CSV файла: {csv_file}")

        try:
            df = pd.read_csv(
                filepath,
                encoding='utf-8-sig',
                parse_dates=['date'],
                converters={
                    'tags': lambda x: ast.literal_eval(x) if isinstance(x, str) and x.strip() else []
                }
            )

            required_columns = ['date', 'text', 'tags', 'summary']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                print(f"В CSV файле отсутствуют колонки: {missing_columns}")
                print(f"Доступные колонки: {list(df.columns)}")
                return []

            print(f"Загружено {len(df)} строк из CSV")

            for idx, row in df.iterrows():
                call_date = pd.to_datetime(row['date'])

                tags = row['tags']
                if isinstance(tags, str):
                    try:
                        tags = eval(tags) if tags.startswith('[') else tags.split(',')
                    except:
                        tags = []

                call_record = {
                    'id': f"call_{idx}",
                    'file_name': csv_file,
                    'call_date': call_date,
                    'year': call_date.year if pd.notna(call_date) else None,
                    'month': call_date.month if pd.notna(call_date) else None,
                    'day': call_date.day if pd.notna(call_date) else None,
                    'full_text': str(row['text']) if pd.notna(row['text']) else '',
                    'summary': row.get('summary', '') if 'summary' in df.columns else '',
                    'tags': tags if isinstance(tags, list) else [tags],
                    'text_length': len(str(row['text'])) if pd.notna(row['text']) else 0,
                    'source_file': filepath,
                    'drive_path': self.drive_path if self.drive_path else None
                }

                all_calls.append(call_record)

                if limit and idx + 1 >= limit:
                    break

            self.calls_cache = all_calls

            print(f" Преобразовано {len(all_calls)} записей звонков")

            if all_calls:
                dates = [c['call_date'] for c in all_calls if c['call_date']]
                if dates:
                    min_date = min(dates)
                    max_date = max(dates)
                    print(f" Диапазон дат: {min_date.strftime('%d.%m.%Y')} - {max_date.strftime('%d.%m.%Y')}")

                all_tags = []
                for call in all_calls:
                    all_tags.extend(call['tags'])
                unique_tags = set(all_tags)
                print(f"Уникальных тегов: {len(unique_tags)}")

            return all_calls

        except Exception as e:
            print(f"Ошибка чтения CSV файла {csv_file}: {e}")
            return []

    def _extract_date_from_filename(self, filename: str) -> datetime:
        patterns = [
            r'(\d{4})-(\d{2})-(\d{2})',  # YYYY-MM-DD
            r'(\d{2})\.(\d{2})\.(\d{4})',  # DD.MM.YYYY
            r'(\d{4})(\d{2})(\d{2})',  # YYYYMMDD
        ]

        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                groups = match.groups()
                if len(groups) == 3:
                    if pattern == patterns[0]:  # YYYY-MM-DD
                        year, month, day = map(int, groups)
                        return datetime(year, month, day)
                    elif pattern == patterns[1]:  # DD.MM.YYYY
                        day, month, year = map(int, groups)
                        return datetime(year, month, day)
                    elif pattern == patterns[2]:  # YYYYMMDD
                        year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
                        return datetime(year, month, day)

        filepath = os.path.join(self.csv_dir, filename)
        return datetime.fromtimestamp(os.path.getmtime(filepath))

    def setup_in_memory_db(self):
        """Creates in-memory SQLite for fast queries"""
        if self.conn is not None:
            return self.conn

        self.conn = sqlite3.connect(':memory:')
        cursor = self.conn.cursor()

        cursor.execute("""
        CREATE TABLE calls (
            id TEXT PRIMARY KEY,
            file_name TEXT,
            call_date TEXT,
            year INTEGER,
            month INTEGER,
            day INTEGER,
            full_text TEXT,
            summary TEXT,
            tags_json TEXT,
            text_length INTEGER,
            source_file TEXT,
            drive_path TEXT
        )
        """)

        cursor.execute("""
        CREATE TABLE call_tags (
            call_id TEXT,
            tag TEXT,
            FOREIGN KEY (call_id) REFERENCES calls(id)
        )
        """)

        calls = self.load_all_calls()
        for call in calls:
            cursor.execute("""
            INSERT INTO calls (id, file_name, call_date, year, month, day, 
                              full_text, summary, tags_json, text_length, source_file, drive_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                call['id'],
                call['file_name'],
                call['call_date'].isoformat(),
                call['year'],
                call['month'],
                call['day'],
                call['full_text'],
                call['summary'],
                json.dumps(call['tags'], ensure_ascii=False),
                call['text_length'],
                call['source_file'],
                call.get('drive_path', '')
            ))

            for tag in call['tags']:
                cursor.execute(
                    "INSERT INTO call_tags (call_id, tag) VALUES (?, ?)",
                    (call['id'], tag)
                )

        self.conn.commit()

        source = "Google Drive" if self.drive_path else "локальной папки"
        print(f" Данные загружены в in-memory SQLite ({len(calls)} записей из {source})")
        return self.conn

    @contextmanager
    def get_cursor(self):
        if self.conn is None:
            self.setup_in_memory_db()

        cursor = self.conn.cursor()
        try:
            yield cursor
        finally:
            cursor.close()



class Planner:
    def __init__(self, model, datasphere_node_url=None, client=None, drive_path=None, config_path='config.yml'):
        self.is_local = False  #isinstance(model, Llama)

        self.drive_path = drive_path
        self.timeout = 600
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        self.available_tags = config.get('tags_list', [])
        self.client = client
        self.model_name = model

    def create_analysis_plan(self, user_query: str, query_history: [] = None) -> AnalysisPlan:
        prompt = self._build_planner_prompt(user_query, query_history)

        if self.is_local:
            response = self.model(
                prompt,
                max_tokens=500,
                temperature=0.1)
        else:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={'temperature': 0.1, 'timeout': self.timeout, 'num_ctx': 4096}
            )
        cleaned = re.sub(r'^```json\s*', '', response['response'])   # remove ```json in the beginning
        cleaned = re.sub(r'\s*```$', '', cleaned)

        plan_data = json.loads(cleaned)
        try:
            plan_data = json.loads(cleaned)
        except:
            raise
        print('Plan: ', plan_data)

        time_period = self._parse_time_period(plan_data.get('time_period', {}))

        target_tags = self._validate_tags(plan_data.get('target_tags', []))

        metrics = self._parse_metrics(plan_data.get('metrics', []))

        keywords = plan_data.get('keywords', [])

        keyword_metrics = self._parse_metrics(plan_data.get('keyword_metrics', []))

        semantic_metrics = self._parse_metrics([plan_data.get('semantic_metrics', '')])
        print('semantic metrics:', semantic_metrics)

        return AnalysisPlan(
            time_period=time_period,
            target_tags=target_tags,
            metrics=metrics,
            keywords=keywords,
            keyword_metrics=keyword_metrics,
            grouping=plan_data.get('grouping', 'month'),
            semantic_queries=[user_query],
            semantic_metrics=semantic_metrics
        )

    def _build_planner_prompt(self, user_query: str, query_history: [] = None) -> str:
        current_date = datetime.now().strftime("%Y-%m-%d")        
        tags = ', '.join(self.available_tags)
        inject = ''
        # if query_history:
        #     n = len(query_history)
        #     n = min(n, 3)
        #     queries = ''
        #     for i in range(n):
        #         queries = queries.join(query_history[-n-1]['query'])+'; '
        #     inject = f'ПРОЧТИ ПРЕДЫДУЩИЕ ЗАПРОСЫ (ты уже ответил на них ранее!), ЕСЛИ КОНТЕКСТ НЕОБХОДИМ ТЕБЕ ДЛЯ ПОНИМАНИЯ НОВОГО ЗАПРОСА: "{queries}".'
        # else:
        #     inject = ''


        return f"""Ты — аналитик базы телефонных звонков и писем компании по аренде ковров.
Клиентами являются любые юр. лица (магазины, банки, больницы, аптеки, театры и так далее)

ЗАПРОС ТВОЕГО ПОЛЬЗОВАТЕЛЯ: "{user_query}".
{inject}

ТВОЯ ЗАДАЧА: Создать план анализа.
Система будет обращаться по твоему плану к текстам с записями телефонных звонков и писем клиентов за несколько последних лет, содержащими описательные теги каждого звонка.

ДОСТУПНЫЕ ТЕГИ:
{tags}

МЕТРИКИ, которые система может посчитать для тебя для ответа на запрос, если это необходимо:
1. count_by_tag - подсчет звонков с заданным тегом за период
2. top_n_tags - самые частые теги звонков за период
3. tag_trends - система сгруппирует подсчет тегов по месяцам, неделям или дням, в зависимости от твоей инструкции. Например, чтобы увидеть динамику встречаемости тега за год или полгода, лучше попроси группировать по месяцам, а чтобы посмотреть динамику за неделю, - по дням. Ты получишь массив с встречаемостью тега в каждой группе, с указанием дат. Например, при группировке по месяцам, ты увидишь даты начала и конца каждого месяца и соответствующее ему число тегов.

Также каждый звонок имеет ровно один тег "call", а каждое сообщение почты - тег "mail". Он нужен, если необходимо посчитать количество всех звонков или писем за какой-либо период. Если твоего пользователя интересует число звонков независимо от их содержания, используй тег "call" для их группировки и / или подсчета.

Только если доступных тегов не достаточно для ответа на специфический вопрос пользователя, система может обратиться к кратким содержаниям звоноков и писем и подсчитать для тебя количество релевантных текстов.
Если ты считаешь, что тегов не достаточно, используй эту опцию. Для этого в твоем ответе выбери одну метрику, которую хочешь посчитать на основе числа релевантных текстов с группировкой по периодам:
4. count_by_semantic
или
5. semantic_trends

Сегодняшняя дата: {current_date} - используй ее, чтобы правильно определить временной период из запроса в случае, если в запросе временной период указан относительно сегодняшнего дня (например, "в прошлом году" и т.п.)

ВЕРНИ JSON с планом того, что системе нужно извлечь из данных для ответа на запрос, а именно: за какой период понадобятся данные? По каким именно тегам выбирать данные для ответа на данный запрос? Какие метрики подсчитать по этим данным для ответа на данный запрос?
{{
  "time_period": {{
    "description": "описание периода",
    "start": "YYYY-MM-DD или null",
    "end": "YYYY-MM-DD или null"
  }},
  "target_tags": ["тег1", "тег2", ... (1 or more tags)],
  "metrics": ["count_by_tag" and/or "tag_trends" and/or "top_n_tags" (necessary metrics)],
  "semantic_metrics": "count_by_semantic" or "semantic_trends",
  "grouping": "month/week/day"
}}
Верни только JSON!
Ответ:
"""

    def _parse_time_period(self, period_data: Dict) -> Dict[str, Any]:
        today = datetime.now()
        description = period_data.get('description', '')

        start = today - timedelta(days=30)
        end = today

        if period_data.get('start'):
            try:
                start = datetime.fromisoformat(period_data['start'])
            except:
                pass
        if period_data.get('end'):
            try:
                end = datetime.fromisoformat(period_data['end'])
            except:
                pass

        return {
            'start': start,
            'end': end,
            'description': description or f"с {start.strftime('%d.%m.%Y')} по {end.strftime('%d.%m.%Y')}"
        }

    def _validate_tags(self, tags: List[str]) -> List[str]:
        valid_tags = []
        for tag in tags:
            for available_tag in self.available_tags:
                if tag.lower() in available_tag.lower() or available_tag.lower() in tag.lower():
                    valid_tags.append(available_tag)
                    break

        return valid_tags or ['низкое качество стирки или чистки']

    def _parse_metrics(self, metrics: List[str] | None) -> List[MetricType]:
        if not metrics:
            return []

        metric_map = {
            'count_by_tag': MetricType.COUNT_BY_TAG,
            'top_n_tags': MetricType.TOP_N_TAGS,
            'tag_trends': MetricType.TAG_TRENDS,
            'comparison': MetricType.COMPARISON,
            'count_by_keyword': MetricType.COUNT_BY_TAG,
            'keyword_trends': MetricType.TAG_TRENDS,
            'count_by_semantic': MetricType.COUNT_BY_SEMANTIC,
            'semantic_trends': MetricType.SEMANTIC_TRENDS
        }

        result = []
        for metric in metrics:
            if metric in metric_map:
                result.append(metric_map[metric])

        return result or [MetricType.COUNT_BY_TAG]



class QueryExecutor:
    def __init__(self, data_loader: DriveDataLoader, model, client):
        self.client = client
        self.data_loader = data_loader
        self.model_name = model

    def execute_plan(self, plan: AnalysisPlan) -> Dict[str, Any]:
        if len(plan.target_tags) == 0 and len(plan.keywords) == 0 and len(plan.semantic_queries) == 0:
            return {}
        results = {}

        all_calls = self.data_loader.load_all_calls()

        if not all_calls:
            print("Нет данных для анализа")
            return {
                'error': 'Нет данных для анализа',
                'summary_stats': {
                    'total_calls': 0,
                    'period': plan.time_period['description'],
                    'date_range': f"{plan.time_period['start'].strftime('%Y-%m-%d')} - {plan.time_period['end'].strftime('%Y-%m-%d')}"
                }
            }

        print(f'{len(all_calls)} звонков всего')
        filtered_calls = self._filter_calls_by_period(all_calls, plan.time_period)
        print(f'{len(filtered_calls)} звонков после фильтрации по периоду')

        for metric in plan.metrics:
            if metric == MetricType.COUNT_BY_TAG:
                results['count_by_tag'] = self._count_by_tag(filtered_calls, plan.target_tags)

            elif metric == MetricType.TAG_TRENDS:
                results['tag_trends'] = self._tag_trends(
                    filtered_calls,
                    plan.target_tags,
                    plan.grouping
                )

            elif metric == MetricType.TOP_N_TAGS:
                results['top_n_tags'] = self._top_n_tags(filtered_calls, n=5)

            elif metric == MetricType.COMPARISON:
                results['comparison'] = self._compare_tags(
                    filtered_calls,
                    plan.comparison_tags or plan.target_tags[:2]
                )
    
        if plan.keywords and plan.keyword_metrics:
            for metric in plan.keyword_metrics:
                if metric == MetricType.COUNT_BY_TAG:  # count_by_keyword
                    results['count_by_keyword'] = self._count_by_keyword(filtered_calls, plan.keywords)
                elif metric == MetricType.TAG_TRENDS:  # keyword_trends
                    results['keyword_trends'] = self._keyword_trends(
                        filtered_calls,
                        plan.keywords,
                        plan.grouping
                    )

        print(plan)
        if plan.semantic_queries and plan.semantic_metrics:
            print('Start semantic analysis...')
            for metric in plan.semantic_metrics:
                for query in plan.semantic_queries:
                    result_key = f"semantic_{query.replace(' ', '_')[:30]}"  # ключ для результата
                    
                    if metric == MetricType.COUNT_BY_SEMANTIC:
                        result = self._count_by_semantic_simple(filtered_calls, query)
                        results[result_key] = result
                        
                    elif metric == MetricType.SEMANTIC_TRENDS:
                        result = self._classify_by_semantic_query(
                            filtered_calls, 
                            query, 
                            grouping=plan.grouping
                        )
                        results[result_key] = result

        results['summary_stats'] = {
            'total_calls': len(filtered_calls),
            'period': plan.time_period['description'],
            'date_range': f"{plan.time_period['start'].strftime('%Y-%m-%d')} - {plan.time_period['end'].strftime('%Y-%m-%d')}",
            'data_source': 'Google Drive' if self.data_loader.drive_path else 'Local'
        }

        return results

    def _filter_calls_by_period(self, calls: List[Dict], period: Dict) -> List[Dict]:
        start_date = period['start']
        end_date = period['end']

        filtered = []
        for call in calls:
            call_date = call['call_date']
            call_date = call_date.replace(tzinfo=None) if call_date.tzinfo else call_date
            if start_date <= call_date <= end_date:
                filtered.append(call)

        return filtered

    def _count_by_tag(self, calls: List[Dict], target_tags: List[str]) -> Dict[str, int]:
        counts = defaultdict(int)

        for call in calls:
            for tag in call['tags']:
                for target in target_tags:
                    if target.lower() in tag.lower() or tag.lower() in target.lower():
                        counts[target] += 1
                        break

        return dict(counts)

    def _tag_trends(self, calls: List[Dict], target_tags: List[str], grouping: str) -> Dict[str, List]:
        if not target_tags or not calls:
            return {}

        trends = defaultdict(lambda: defaultdict(int))

        for call in calls:
            if grouping == 'month':
                period_key = call['call_date'].strftime('%Y-%m')
            elif grouping == 'week':
                year, week, _ = call['call_date'].isocalendar()
                period_key = f"{year}-W{week:02d}"
            else:  # day
                period_key = call['call_date'].strftime('%Y-%m-%d')

            for tag in call['tags']:
                for target in target_tags:
                    if target.lower() in tag.lower() or tag.lower() in target.lower():
                        trends[target][period_key] += 1
                        break

        result = {}
        for tag, period_counts in trends.items():
            result[tag] = [
                {'period': period, 'count': count}
                for period, count in sorted(period_counts.items())
            ]

        return result

    def _top_n_tags(self, calls: List[Dict], n: int = 5) -> List[Dict]:
        tag_counter = Counter()

        for call in calls:
            tag_counter.update(call['tags'])

        return [
            {'tag': tag, 'count': count}
            for tag, count in tag_counter.most_common(n)
        ]

    def _compare_tags(self, calls: List[Dict], tags: List[str]) -> Dict[str, Any]:
        if len(tags) < 2:
            tags = tags + [None] * (2 - len(tags))

        counts = self._count_by_tag(calls, tags[:2])

        return {
            'tag1': {'name': tags[0], 'count': counts.get(tags[0], 0)},
            'tag2': {'name': tags[1], 'count': counts.get(tags[1], 0)},
            'total_calls': len(calls),
            'ratio': counts.get(tags[0], 0) / counts.get(tags[1], 1) if counts.get(tags[1], 0) > 0 else 0
        }

    def _count_by_keyword(self, calls: List[Dict], keywords: List[str]) -> Dict[str, int]:
        counts = defaultdict(int)
    
        for call in calls:
            summary = call.get('summary', '').lower()
            for keyword in keywords:
                if keyword.lower() in summary:
                    counts[keyword] += 1
    
        return dict(counts)

    def _keyword_trends(self, calls: List[Dict], keywords: List[str], grouping: str) -> Dict[str, List]:
        if not keywords or not calls:
            return {}

        trends = defaultdict(lambda: defaultdict(int))

        for call in calls:
            if grouping == 'month':
                period_key = call['call_date'].strftime('%Y-%m')
            elif grouping == 'week':
                year, week, _ = call['call_date'].isocalendar()
                period_key = f"{year}-W{week:02d}"
            else:
                period_key = call['call_date'].strftime('%Y-%m-%d')

            summary = call.get('summary', '').lower()
            for keyword in keywords:
                if keyword.lower() in summary:
                    trends[keyword][period_key] += 1
    
        result = {}
        for keyword, period_counts in trends.items():
            result[keyword] = [
                {'period': period, 'count': count}
                for period, count in sorted(period_counts.items())
            ]
    
        return result

    def _count_by_semantic_simple(self, calls: List[Dict], user_query: str) -> Dict:
        result = self._classify_by_semantic_query(calls, user_query, grouping='month')
        return {
            'query': user_query,
            'total_relevant': result['total_relevant'],
            'total_calls': result['total_calls'],
            'percentage': result['percentage'],
            'examples': result['examples'][:5]
        }

    def _classify_by_semantic_query(self, calls: List[Dict], user_query: str, 
                                grouping: str = "month", batch_size: int = 500) -> Dict:
        if not calls:
            return {
                'total_relevant': 0,
                'total_calls': 0,
                'percentage': 0,
                'trends': [],
                'examples': []
            }
    
        calls_by_period = defaultdict(list)
        for call in calls:
            call_date = call.get('call_date')
            if call_date:
                if grouping == 'month':
                    period_key = call_date.strftime('%Y-%m')
                elif grouping == 'week':
                    year, week, _ = call_date.isocalendar()
                    period_key = f"{year}-W{week:02d}"
                else:  # day
                    period_key = call_date.strftime('%Y-%m-%d')
                calls_by_period[period_key].append(call)
        
        sorted_periods = sorted(calls_by_period.keys())
        
        period_results = []
        total_relevant = 0
        total_calls = 0
        all_examples = []
        
        print(f"Анализ по периодам ({grouping}): {len(sorted_periods)} периодов")
        
        for period in tqdm(sorted_periods, desc="Анализ по периодам"):
            period_calls = calls_by_period[period]
            
            period_relevant = 0
            period_examples = []
            
            batches = [period_calls[i:i+batch_size] for i in range(0, len(period_calls), batch_size)]
            
            for batch_idx, batch in enumerate(batches):
                calls_text = []
                for idx, call in enumerate(batch):
                    summary = call.get('summary', '')
                    if summary and summary.strip():
                        calls_text.append(f"{idx+1}. {summary.strip()}")
                
                if not calls_text:
                    continue
                
                prompt = self._build_semantic_batch_prompt(user_query, calls_text)
                print('Calling LLM...')
                llm_response = self._call_llm(prompt)
                print('Parsing LLM response...')
                batch_result = self._parse_semantic_batch_response(llm_response, batch)
                print('Updating period relevant texts...')
                period_relevant += batch_result['relevant_count']
                print('Updating period examples...')
                period_examples.extend(batch_result['examples'])
                    
            
            period_results.append({
                'period': period,
                'relevant': period_relevant,
                'total': len(period_calls),
                'percentage': (period_relevant / len(period_calls) * 100) if period_calls else 0
            })
            
            total_relevant += period_relevant
            total_calls += len(period_calls)
            all_examples.extend(period_examples[:5])  # до 5 примеров на период
        
        trends = [
            {
                'period': r['period'],
                'count': r['relevant'],
                'total_calls': r['total'],
                'percentage': round(r['percentage'], 2)
            }
            for r in period_results
        ]
        
        return {
            'query': user_query,
            'total_relevant': total_relevant,
            'total_calls': total_calls,
            'percentage': round((total_relevant / total_calls * 100), 2) if total_calls else 0,
            'trends': trends,
            'examples': all_examples[:10],
            'grouping': grouping
        }


    def _build_semantic_batch_prompt(self, user_query: str, calls_text: List[str]) -> str:
        calls_formatted = "\n".join(calls_text)

        prompt = f"""Ты анализируешь звонки клиентов. Пользователь хочет найти звонки, которые помогут ответить на его запрос.
    Клиентами являются любые юр. лица (магазины, банки, больницы, аптеки, театры и так далее)

    Запрос пользователя: {user_query}

    Для каждого звонка из списка ниже определи, соответствует ли он этому критерию.
    
    Правила:
    1. Учитывай смысл и контекст, а не только ключевые слова.
    2. Если есть сомнения - считай, что НЕ соответствует
    
    ОТВЕТЬ ТОЛЬКО В ФОРМАТЕ JSON:
    {{
    "relevant_count": число_релевантных_звонков_в_этом_батче,
    "examples": [
        {{"number": 1, "date": "дата", "summary": "текст"}},
        ...
    ]
    }}

    В поле "examples" включи ДО 3 примеров релевантных текстов звонков (если они есть). Если релевантных нет, верни пустой список.

    СПИСОК ЗВОНКОВ ДЛЯ АНАЛИЗА:
    {calls_formatted}

    Твой ответ (только JSON):"""

        return prompt

    def _parse_semantic_batch_response(self, llm_response: str, batch: List[Dict]) -> Dict:
        #try:
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(llm_response)
            relevant_count = data.get('relevant_count', 0)
            examples_data = data.get('examples', [])
            examples = []
            for ex in examples_data[:3]:
                #try:
                    number = int(ex.get('number', 0)) - 1
                    if 0 <= number < len(batch):
                        call = batch[number]
                        call_date = call.get('call_date', '')
                        if hasattr(call_date, 'strftime'):
                            call_date = call_date.strftime('%Y-%m-%d')
                        examples.append({
                            'date': call_date,
                            'summary': call.get('summary', '')[:300],
                            'batch_relevant': True
                        })
                #except (ValueError, KeyError, IndexError):
                #    continue

            return {
                'relevant_count': relevant_count,
                'examples': examples
            }
    
        #except (json.JSONDecodeError, ValueError) as e:
        #    print(f"Ошибка парсинга ответа LLM: {e}")
        #    print(f"Ответ LLM: {llm_response[:200]}...")
    
        #    return {
        #        'relevant_count': 0,
        #        'examples': []
        #    }

    def _call_llm(self, prompt: str) -> str:
        response = self.client.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options={'temperature': 0.3, 'num_ctx': 50000}
        )
        print(response)
        return response['response']

class Analyzer:
    def __init__(self, model, datasphere_node_url = None, client = None, drive_path: str = None):
        self.is_local = False # isinstance(model, Llama)
        self.timeout = 600

        if self.is_local:
            self.model_name = 'local'
            self.model = model
        elif datasphere_node_url:
            self.client = ollama.Client(host=datasphere_node_url, timeout=self.timeout)
            self.model_name = 'from_yandex_node'
            print(f"Mode: Yandex DataSphere (node url: {datasphere_node_url})")
        else:
            self.model_name = model
            self.client = client

        self.drive_path = drive_path


    def generate_answer(self, user_query: str, results: Dict, plan: AnalysisPlan) -> str:
        prompt = self._build_analyzer_prompt(user_query, results, plan)
        
        #print(f"DEBUG analyzer_prompt: {prompt}")

        try:
            if self.is_local:
                response = self.model(prompt,
                                      temperature=0.3)
            else:
                response = self.client.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options={'temperature': 0.3, 'num_ctx': 30000}
                )

            return response['response'].strip()

        except Exception as e:
            return (f" Ошибка анализатора: {e}")


    def _build_analyzer_prompt(self, user_query: str, results: Dict, plan: AnalysisPlan) -> str:
        results_str = json.dumps(results, ensure_ascii=False, indent=2, default=str)

        data_source = "Google Drive" if self.drive_path else "локальной базы"

        return f"""Ты — старший аналитик компании по аренде ковров.

ЗАПРОС КЛИЕНТА: "{user_query}"

Для ответа на запрос система выбрала тексты обращений клиентов за нужный период и посчитала нужные метрики.
- Период, которым интересовался клиент: {plan.time_period['description']}
- Подходящие теги, которые выбрала система для выбора обращений для анализа данного запроса: {', '.join(plan.target_tags)}
- Ключевые слова, по которым также собиралась статистика: {', '.join(plan.keywords)}
- Метрики, которые система подсчитала для выполнения данного запроса, на основании текстов обращений, отобранных по этим тегам: {[m.value for m in plan.metrics]}
- Метрики, которые система по этим ключевым словам: {[m.value for m in plan.keyword_metrics]}

Вот результаты, которые выдала система по подсчетам метрик:
{results_str}

ТВОЯ ЗАДАЧА:
1. Проанализировать цифры в этих результатах (если результат не пустой!)
2. Ответить на запрос клиента
3. Говорить конкретно, с цифрами

ФОРМАТ:
- Краткий вывод
- Детальный анализ

Если ты видишь, что система дала тебе пустые метрики, или информации в результатах не достаточно для ответа на запрос клиента, - так и напиши.

ОТВЕТ НА РУССКОМ:"""


class CallAnalyticsMCP:
    def __init__(self, json_directory: str, model, node_url=None, drive_path: str = None):
        self.is_local = False
        self.timeout = 600
        self.drive_path = drive_path
        self.data_loader = DriveDataLoader(json_directory, drive_path)
        self.api_key_ollama = ''

        self.total_calls = len(self.data_loader.load_all_calls())

        if self.total_calls == 0:
            print("  Внимание: Нет данных для анализа")
            if self.drive_path:
                print(f"  Проверьте наличие файлов в Google Drive: {json_directory}")
        else:
            print(f" Загружено {self.total_calls} звонков")

        self.ollama_cloud_url = 'https://ollama.com/'
        self.client = ollama.Client(host=self.ollama_cloud_url, timeout=self.timeout, headers={'Authorization': f'Bearer {self.api_key_ollama}'})
        self.model_name = model  # "mistral-large-3:675b-cloud"    
        try:
            self.client.list()
            print(f"Ollama Cloud подключен, модель: {self.model_name}")
        except Exception as e:
            print(f"Ошибка подключения к Ollama Cloud: {e}")


        self.planner = Planner(model, node_url, self.client, drive_path)
        self.analyzer = Analyzer(model, node_url, self.client, drive_path)
        self.executor = QueryExecutor(self.data_loader, model, self.client)




    def _setup_ollama_client(self):
        try:
            host = "http://localhost:11434"

            if self.drive_path:
                models_cache_dir = os.path.join(self.drive_path, "models_cache")
                os.makedirs(models_cache_dir, exist_ok=True)

            self.client = ollama.Client(host=host, timeout=self.timeout)

            try:
                self.client.list()
                print(f"Ollama подключен, модель: {self.model_name}")
            except Exception as e:
                print(f"Ошибка подключения к Ollama: {e}")
                print("Убедитесь, что Ollama запущен в Colab")

        except ImportError:
            print(" Ollama не установлен")
            raise

    def process_query(self, user_query: str, query_history: [] = None) -> Dict[str, Any]:
        print(f"\n Анализирую запрос: '{user_query}'")

        if self.drive_path:
            print(f" Источник данных: Google Drive")

        print(" Создаю план анализа...")
        analysis_plan = self.planner.create_analysis_plan(user_query, query_history)

        print(f"    Период: {analysis_plan.time_period['description']}, {analysis_plan.time_period['start']}, {analysis_plan.time_period['end']}")
        print(f"    Теги: {', '.join(analysis_plan.target_tags)}")
        print(f"    Метрики: {[m.value for m in analysis_plan.metrics]}")

        print(" Выполняю анализ...")
        analysis_results = self.executor.execute_plan(analysis_plan)

        print(" Формулирую ответ...")
        answer = self.analyzer.generate_answer(user_query, analysis_results, analysis_plan)

        response = {
            'query': user_query,
            'analysis_plan': analysis_plan.to_dict(),
            'raw_results': analysis_results,
            'answer': answer,
            'total_calls_analyzed': analysis_results.get('summary_stats', {}).get('total_calls', 0),
            'processing_time': datetime.now().isoformat(),
            'model_used': self.planner.model_name,
            'data_source': 'Google Drive' if self.drive_path else 'Local'
        }

        self._print_analysis_summary(analysis_results)

        return response

    def _print_analysis_summary(self, results: Dict[str, Any]):
        print("КРАТКАЯ СТАТИСТИКА:")

        if 'summary_stats' in results:
            stats = results['summary_stats']
            print(f" Период: {stats.get('period', 'N/A')}")
            print(f" Проанализировано звонков: {stats.get('total_calls', 0)}")
            print(f" Источник данных: {stats.get('data_source', 'Local')}")

        if 'count_by_tag' in results:
            counts = results['count_by_tag']
            if counts:
                print("\n Количество по тегам:")
                for tag, count in counts.items():
                    print(f"  • {tag}: {count}")
            else:
                print("\n  Нет совпадений по указанным тегам")

        if 'top_n_tags' in results and results['top_n_tags']:
            print("\n Топ теги:")
            for i, item in enumerate(results['top_n_tags'][:3], 1):
                print(f"  {i}. {item['tag']}: {item['count']}")

        if 'tag_trends' in results:
            for tag, trends in results['tag_trends'].items():
                if trends and len(trends) >= 2:
                    first = trends[0]['count']
                    last = trends[-1]['count']
                    change = ((last - first) / first * 100) if first > 0 else 0
                    print(f"\n Динамика '{tag}': {abs(change):.1f}%")

    def get_system_info(self) -> Dict[str, Any]:
        calls = self.data_loader.load_all_calls()

        all_tags = []
        for call in calls:
            all_tags.extend(call['tags'])

        unique_tags = set(all_tags)

        dates = [call['call_date'] for call in calls]

        return {
            'total_calls': len(calls),
            'unique_tags_count': len(unique_tags),
            'date_range': {
                'start': min(dates).isoformat() if dates else None,
                'end': max(dates).isoformat() if dates else None
            },
            'average_text_length': sum(len(c['full_text']) for c in calls) // len(calls) if calls else 0,
            'model': self.planner.model_name,
            'data_source': 'Google Drive' if self.drive_path else 'Local Files',
            'drive_path': self.drive_path if self.drive_path else None
        }

   
