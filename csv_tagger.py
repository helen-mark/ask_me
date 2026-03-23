import json
import ollama
import pandas as pd
import re
from typing import Optional, Dict, Any
import time
import yaml
#from tagging_agent import TaggingAgent


class CsvProcessor:
    def __init__(self, model: str, output_csv_path: str, batch_size: int = 100, mail: bool = False, config_path: str = 'config.yml'):
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)

        self.model = model
        self.output_csv_path = output_csv_path
        self.batch_size = batch_size
        self.tags_list = config.get('tags_list', [])
        self.mail = mail
        self.is_local = False
        self.client = ollama.Client(headers={"Connection": "keep-alive"})
        self.processed_files = set()

    def _run_tagging_locally(self, text: str) -> Dict[str, Any]:
        try:
            return self.get_tags_from_llm(text)
        except Exception as e:
            print(f"Ошибка тегирования: {e}")
            return {'result': [], 'summary': f'Ошибка: {str(e)}'}

    def process(self, input_csv_path: Optional[str] = None) -> None:
        csv_path = input_csv_path if input_csv_path else self.output_csv_path

        df = pd.read_csv(csv_path)

        if 'tags' not in df.columns:
            df['tags'] = None

        mask_empty = ~df['text'].isna() & (df['tags'].isna() | (df['tags'] == ''))
        indices_to_process = df[mask_empty].index.tolist()

        if not indices_to_process:
            print("Все строки уже имеют теги.")
            return

        print(f"Найдено {len(indices_to_process)} строк без тегов.")

        for i in range(0, len(indices_to_process), self.batch_size):
            batch_indices = indices_to_process[i:i + self.batch_size]
            batch_df = df.loc[batch_indices]

            print(f"Обработка батча {i // self.batch_size + 1}, размер: {len(batch_indices)}")

            for idx in batch_indices:
                text = df.loc[idx, 'text']
                if pd.isna(text) or text == '':
                    df.at[idx, 'tags'] = '[]'
                    df.at[idx, 'summary'] = 'нет'
                    continue

                try:
                    result = self._run_tagging_locally(text)

                    if 'result' in result:
                        tags = result['result']
                        df.at[idx, 'tags'] = str(tags)
                    else:
                        df.at[idx, 'tags'] = '[]'
                    if 'summary' in result:
                        s = result['summary']
                        df.at[idx, 'summary'] = str(s)
                    else:
                        df.at[idx, 'summary'] = 'нет'

                except Exception as e:
                    print(f"Ошибка обработки индекса {idx}: {e}")
                    df.at[idx, 'tags'] = '[]'
                    df.at[idx, 'summary'] = 'нет'

                time.sleep(0.1)

            df.to_csv(self.output_csv_path, index=False)
            print(f"Батч {i // self.batch_size + 1} обработан и сохранен.")

        print("Обработка завершена.")


    def get_tags_from_llm(self, text: str) -> Dict[str, Any]:
        truncated_text = text[:3000] + "..." if len(text) > 3000 else text

        prompt = f"""Ты — специалист по категоризации телефонных разговоров.
Есть записи телефонных разговоров менеджеров с клиентами, которые берут в аренду грязезащитные ковры и получают услуги по их доставке (замене) и чистке.

Вот текст одного разговора:
{truncated_text}

ТВОЕ ЗАДАНИЕ: Во-первых, Проанализируй этот разговорный текст и верни краткое summary, характеризующие 1-2 главных причины обращения клиента.

Во-вторых, Проанализируй этот разговорный текст. Ознакомься со списком заранее сформированных описаний: 
{', '.join(self.tags_list)}. Подходят ли какие-нибудь из них этому тексту?
Присвой тексту от 0 до 3 описаний из фиксированного списка, только если они действительно хорошо характеризуют причины обращения клиента.
Например, клиент долго не получает ответ на его заявку о том, что ему не доставили вовремя ковер. Тогда присвой два описания: про долгое ожидание ответа и про недоставку (несвоевременную замену).
Либо клиент хочет возобновить услуги И при этом добавить больше ковров, чем было у него раньше. Тогда подойдет описание про возобновление услуг и описание про добавление ковров. И так далее.

Если клиент выражает недовольство ценами или непонимание, почему цены неожиданно выросли, - выбирай описание "клиент недоволен ценами". Но если клиент просто запрашивает информацию о планируемом росте цен, или уточняет, когда будет индексация, не выражая непонимания или недовольства, - то выбирай описание, связанное с уточнением деталей. Сам факт роста цен в связи с инфляцией - нормален.
Аналогично с другими проблемами: если клиент упоминает ключевые слова, связанные с какой-либо проблемой, - это не всегда означает факт возникновения проблемы. Будь внимателен!

Выбирай описание "консультация или уточнение деталей" только в случае, если нет никакой другой причины обращения!
Если ни одно описание не подходит, - просто не присваивай никаких описаний.

ВЕРНИ ОТВЕТ ТОЛЬКО В ФОРМАТЕ JSON:
{{
  "result": ["описание1", "описание2"],
  "summary": "причины обращения клиента своими словами"
}}
Если текст не содержит ясной причины обращения - верни пустой список описаний и слово "нет" в качестве summary.
"""

        prompt_mail = f"""Ты — специалист по категоризации писем электронной почты.
Есть емейл сообщения от клиентов, которые берут в аренду грязезащитные ковры и получают услуги по их доставке (замене) и чистке.

Вот один текст:
{truncated_text}

ТВОЕ ЗАДАНИЕ: Во-первых, Проанализируй этот разговорный текст и верни краткое summary, характеризующие 1-2 главных причины обращения клиента.

Во-вторых, Проанализируй этот текст. Ознакомься со списком заранее сформированных описаний: 
{', '.join(self.tags_list)}. Подходят ли какие-нибудь из них этому тексту?
Присвой тексту от 0 до 3 описаний из фиксированного списка, только если они действительно хорошо характеризуют причины обращения клиента. 
Например, клиент долго не получает ответ на его заявку о том, что ему не доставили вовремя ковер. Тогда присвой два описания: "долго нет ответа на заявку" и "не заменили ковры вовремя".
Либо клиент хочет возобновить услуги И при этом добавить больше ковров, чем было у него раньше. Тогда подойдет описание про возобновление услуг и описание про добавление ковров. И так далее.

Если клиент выражает недовольство ценами или непонимание, почему цены неожиданно выросли, - выбирай описание "клиент недоволен ценами". Но если клиент просто запрашивает информацию о планируемом росте цен, или уточняет, когда будет индексация, не выражая непонимания или недовольства, - то выбирай описание, связанное с уточнением деталей. Сам факт роста цен в связи с инфляцией - нормален.
Аналогично с другими проблемами: если клиент упоминает ключевые слова, связанные с какой-либо проблемой, - это не всегда означает факт возникновения проблемы. Будь внимателен!

Выбирай описание "консультация или уточнение деталей" только в случае, если нет никакой другой причины обращения!
Если ни одно описание не подходит, - просто не присваивай никаких описаний. Не придумывай описания! Используй только данный тебе список.

ВЕРНИ ОТВЕТ ТОЛЬКО В ФОРМАТЕ JSON:
{{
  "result": ["описание1", "описание2"],
  "summary": "причины обращения клиента своими словами"
}}
Если текст не содержит ясной причины обращения - верни пустой список описаний и слово "нет" в качестве summary.
"""
    # print(prompt_mail)
    #         import psutil
    #         import GPUtil

    #         def check_resources():
    #             # CPU и RAM
    #             print(f"CPU: {psutil.cpu_percent()}%")
    #             print(f"RAM: {psutil.virtual_memory().percent}%")

    #             # GPU
    #             try:
    #                 gpus = GPUtil.getGPUs()
    #                 for gpu in gpus:
    #                     print(f"GPU {gpu.name}: {gpu.memoryUtil*100:.1f}% used")
    #             except:
    #                 pass
    #         check_resources()

        empty_response = {
            "result": [],
            "summary": '',
            "additional_tags": [],
            "reasoning": "Ошибка"
        }

        try:
            if len(truncated_text) < 30:
                print('Text is too short')
                return empty_response
            # if self.is_local:
            #     response = self.model(prompt_mail if self.mail else prompt,
            #                           temperature=0.3,
            #                           top_p=0.9,
            #                           # num_gpu=-1,
            #                           num_ctx=4096)
            # else:
            print('Getting response ...')
            response = self.client.generate(
                model=self.model,
                prompt=prompt_mail if self.mail else prompt,
                keep_alive=-1,
                options={
                    'temperature': 0.3,
                    'top_p': 0.9,
                    # "num_gpu": -1,
                    'num_ctx': 4096
                }
            )

            response_text = response['response']
            print('response + ctx8: ', response_text)
            token_count = response['prompt_eval_count']
            print(f"Real n tokens in prompt: {token_count}")

            prompt_selfcheck = f"""Ты — специалист по категоризации телефонных разговоров.
Есть записи телефонных разговоров менеджеров с клиентами, которые берут в аренду грязезащитные ковры и получают услуги по их доставке (замене) и чистке.

Вот текст одного разговора:
{truncated_text}

Ты присвоил ему следующие теги:
{response_text}, которые ты выбрал из списка:
{', '.join(self.tags_list)}

Проверь себя! Точно ли каждый из выбранных тобой тегов отражает реальную проблему / причину обращения клиента, а не просто содержит те же ключевые слова, что встречаются в тексте разговора?
Не забыл ли ты добавить какие-нибудь теги?
Если нужно - исправь свой ответ. Если не нашел неточностей - оставь ответ тем же.
Выбирай тег консультация_или_уточнение_деталей, ТОЛЬКО если нет никакой другой причины обращения!

ВЕРНИ ОТВЕТ ТОЛЬКО В ФОРМАТЕ JSON (от 0 до 3 тегов):
{{
  "result": ["tag1", "tag2"]
}}
Если текст не содержит ясной причины обращения - верни пустой json
"""

        #             print('Getting response - try 2 ...')
        #             response = self.client.generate(
        #                 model=self.model_name,
        #                 prompt=prompt_selfcheck,
        #                 options={
        #                     'temperature': 0.3,
        #                     'top_p': 0.8,
        #                     'num_ctx': 3000
        #                 }
        #             )
        #             response_text = response['response']
        #             print('response (corrected): ', response_text)

            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())

                valid_selected = []
                for tag in result.get('result', []):
                    if tag in self.tags_list:
                        valid_selected.append(tag)
                    else:
                        print(f"Модель придумала тег '{tag}', игнорирую")

                result['result'] = valid_selected

                return result
            else:
                raise ValueError("LLM не вернула JSON")

        except Exception as e:
            print(f"    Ошибка при запросе к LLM: {e}")
            return empty_response