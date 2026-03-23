import json
import os
from datetime import datetime
import ollama
from pathlib import Path
import mcp_orchestrator


def enhanced_interactive_mode(_model, node_url = None, csv_dir: str = None, results_dir: str = None, drive_path: str = None):
    def show_help(in_drive_mode: bool):
        help_text = """
     КОМАНДЫ:

    АНАЛИТИЧЕСКИЕ ЗАПРОСЫ:
      Введите ваш вопрос, например:
      • "Сколько жалоб на качество в этом месяце?"
      • "Какие самые частые темы обращений?"

    СИСТЕМНЫЕ КОМАНДЫ:
      /? или /помощь      - справка
      /выход             - завершить работу
      /статистика        - статистика данных
      /история           - история запросов
      /сохранить         - сохранить последний результат
    """
        print(help_text)

    def show_system_stats(system, in_drive_mode: bool):
        info = system.get_system_info()

        print("\n СТАТИСТИКА СИСТЕМЫ:")

        if in_drive_mode:
            print(" Режим: Google Drive")

        print(f" Всего звонков: {info['total_calls']}")
        print(f"  Уникальных тегов: {info['unique_tags_count']}")

        if info['date_range']['start']:
            start_date = datetime.fromisoformat(info['date_range']['start']).strftime('%d.%m.%Y')
            end_date = datetime.fromisoformat(info['date_range']['end']).strftime('%d.%m.%Y')
            print(f" Период данных: {start_date} - {end_date}")

        print(f" Средняя длина текста: {info['average_text_length']} симв.")
        print(f" Модель: {info['model']}")
        print(f" Источник: {info['data_source']}")

        if 'drive_path' in info and info['drive_path']:
            print(f" Google Drive путь: {info['drive_path']}")

    def show_query_history(history):
        if not history:
            print(" История запросов пуста")
            return

        print("\n ИСТОРИЯ ЗАПРОСОВ:")

        for i, item in enumerate(reversed(history[-10:]), 1):
            time_str = item['timestamp'].strftime('%H:%M')

            query_preview = item['query']
            if len(query_preview) > 50:
                query_preview = query_preview[:47] + "..."

            print(f"{i}. [{time_str}] {query_preview}")

            if item.get('processing_time'):
                print(f"   ⏱️  {item['processing_time']:.1f} сек")

        print("-" * 60)
        print(f"Всего запросов: {len(history)}")

    def save_last_result(history, results_dir):
        if not history:
            print("Нет результатов для сохранения")
            return

        completed_queries = [h for h in history if h.get('status') == 'completed' and 'result' in h]

        if not completed_queries:
            print("Нет завершенных запросов для сохранения")
            return

        last_result = completed_queries[-1]['result']

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_query = last_result['query'][:30].replace(' ', '_').replace('?', '').replace('/', '_')
        filename = f"result_{timestamp}_{safe_query}.json"
        filepath = os.path.join(results_dir, filename)

        os.makedirs(results_dir, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(last_result, f, ensure_ascii=False, indent=2)

        print(f"Результат сохранен в: {filepath}")

    print("""
╔══════════════════════════════════════════╗
║      АНАЛИТИК ЗВОНКОВ v3.1               ║
║      Google Drive Edition                ║
╚══════════════════════════════════════════╝
    """)

    JSON_DIRECTORY = csv_dir if csv_dir else "csv_calls"
    RESULTS_DIRECTORY = results_dir if results_dir else "saved_results"

    if not os.path.exists(JSON_DIRECTORY):
        print(f"Директория {JSON_DIRECTORY} не найдена!")
        print("Сначала добавьте csv файл в директорию csv_calls/")

        return

    os.makedirs(RESULTS_DIRECTORY, exist_ok=True)

    system = mcp_orchestrator.CallAnalyticsMCP(JSON_DIRECTORY, _model, node_url, drive_path)

    query_history = []

    while True:
        try:
            prompt = f"\nВопрос (/? для помощи): "
            user_input = input(prompt).strip()

            if user_input.lower() in ['/выход', '/exit', 'выход', 'exit', '/q', 'q']:
                break

            elif user_input.lower() in ['/?', '/помощь', '/help']:
                show_help(False)
                continue

            elif user_input.lower() == '/статистика':
                show_system_stats(system, False)
                continue

            elif user_input.lower() == '/история':
                show_query_history(query_history)
                continue
            #
            # elif user_input.lower() == '/очистить':
            #     os.system('cls' if os.name == 'nt' else 'clear')
            #     print(" Экран очищен")
            #     continue

            elif user_input.lower().startswith('/сохранить'):
                save_last_result(query_history, RESULTS_DIRECTORY)
                continue

            elif not user_input:
                continue

            print(f"Анализирую: '{user_input}'")

            query_history.append({
                'query': user_input,
                'timestamp': datetime.now(),
                'status': 'processing',
                'mode': 'local'
            })

            if len(query_history) > 20:
                query_history = query_history[-20:]

            start_time = datetime.now()
            result = system.process_query(user_input, query_history)
            processing_time = (datetime.now() - start_time).total_seconds()

            query_history[-1]['status'] = 'completed'
            query_history[-1]['result'] = result
            query_history[-1]['processing_time'] = processing_time

            print(f"ОТВЕТ ({processing_time:.1f} сек):")
            print(result['answer'])

            print(f"Проанализировано звонков: {result.get('total_calls_analyzed', 0)}")

            print("\n Быстрые действия:")
            print("  • Задать уточняющий вопрос")
            print("  • /сохранить - сохранить этот результат")
            print("  • /история - показать предыдущие запросы")
            print("  • /выход - завершить работу")

        except KeyboardInterrupt:
            print("\n\nЗавершаю работу...")
            break
        except Exception as e:
            print(f"\n Ошибка: {e}")
            if query_history:
                query_history[-1]['status'] = 'error'
                query_history[-1]['error'] = str(e)

