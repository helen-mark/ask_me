import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import json
import yaml
import sys
import os
import hmac
import mcp_orchestrator

CONFIG_PATH='config.yml'
CREDENTIALS_PATH='credentials.yml'

def check_password():
    def password_entered():
        if hmac.compare_digest(
            st.session_state["password"],
            st.secrets["password"]
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # dont store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input(
            "Введите пароль",
            type="password",
            on_change=password_entered,
            key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        st.text_input(
            " Введите пароль",
            type="password",
            on_change=password_entered,
            key="password"
        )
        st.error("❌ Неверный пароль")
        return False
    else:
        return True

if not check_password():
    st.stop()

st.set_page_config(
    page_title="Аналитик почты",
    page_icon="",
    layout="wide"
)

st.set_page_config(
    page_title="Аналитика обращений клиентов",
    layout="wide"
)

#if st.button("Обновить данные"):
#    st.cache_data.clear()
#    st.rerun()

try:
    with open(CONFIG_PATH, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
except:
    config = {
        'folders': {'csv_mail': '.', 'saved_results': './results'},
        'llm_model': 'gpt-3.5-turbo'
    }

if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.messages = []
    st.session_state.query_history = []
    st.session_state.system = None


def init_system():
    if not st.session_state.initialized:
        with st.spinner("Загрузка системы аналитики..."):
            try:
                st.session_state.system = mcp_orchestrator.CallAnalyticsMCP(
                    CONFIG_PATH,
                    CREDENTIALS_PATH,
                    config['folders']['csv_mail'],
                    config.get("llm_model", "gpt-3.5-turbo")
                )
                st.session_state.initialized = True
                st.success("Система аналитики готова к работе!")
            except Exception as e:
                st.error(f"Ошибка инициализации: {e}")


def load_data(data_path):
    for f in os.listdir(data_path):
        file_path = os.path.join(data_path, f)
        break
    print(file_path)
    df = pd.read_csv(file_path, encoding='utf-8')
    df = df.sort_values('date', ascending=False).reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date_str'])
    if df['date'].dt.tz is not None:
        df['date'] = df['date'].dt.tz_localize(None)
    st.sidebar.write(f"Диапазон дат: {df['date'].min()} - {df['date'].max()}")

    if 'summary' in df.columns:
        df['summary'] = df['summary'].fillna('нет')

    if 'tags' in df.columns:
        df['tags'] = df['tags'].fillna('[]')
        df['tags'] = df['tags'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    else:
        df['tags'] = [[] for _ in range(len(df))]

    return df


def filter_by_timeframe(df, timeframe):
    last_date = datetime.now()  #df['date'].max()

    if timeframe == 'Последний месяц':
        start_date = last_date - timedelta(days=30)
    elif timeframe == 'Последний квартал':
        start_date = last_date - timedelta(days=90)
    elif timeframe == 'Последний год':
        start_date = last_date - timedelta(days=365)
    else:
        return df

    return df[df['date'] >= start_date]


def prepare_tag_data(df, tags_of_interest):
    tag_data = []
    df['week'] = df['date'].dt.to_period('W-MON').dt.start_time
    for tag in tags_of_interest:
        tag_counts = df[df['tags'].apply(lambda x: tag in x)].groupby(
            'week'
        ).size().reset_index()

        if not tag_counts.empty:
            tag_counts.columns = ['date', 'count']
            tag_counts['tag'] = tag
            tag_data.append(tag_counts)

    if tag_data:
        return pd.concat(tag_data, ignore_index=True)
    return pd.DataFrame(columns=['date', 'count', 'tag'])


def get_recent_records_by_tag(df, tag_names, search_in_summary=False, n_records=500):
    if search_in_summary:
        column = 'summary'
    else:
        column = 'tags'
        
    filtered_df = df[df[column].apply(lambda x: any(tag_name in x for tag_name in tag_names))].copy()

    if filtered_df.empty:
        return pd.DataFrame()

    filtered_df = filtered_df.sort_values('date', ascending=False).head(n_records)

    result_df = pd.DataFrame()
    result_df['is_read'] = filtered_df['is_read']

    result_df['Дата и время'] = filtered_df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')

    if 'from' in filtered_df.columns:
        result_df['Источник'] = filtered_df['from'].fillna('')
    elif 'source_audio' in filtered_df.columns:
        result_df['Источник'] = filtered_df['source_audio'].fillna('')
    else:
        result_df['Источник'] = ''

    result_df['Краткое содержание'] = filtered_df['summary']
    result_df['Теги'] = filtered_df['tags'].apply(lambda x: ', '.join(x) if x else '')

    if 'text' in filtered_df.columns:
        result_df['Исходный текст'] = filtered_df['text'].fillna('')
    print(filtered_df.columns)
    return result_df

st.title("Аналитика обращений клиентов")
st.markdown("---")

left_col, right_col = st.columns([5, 2])

def filter_by_selected_tags(df):
    all_tags = set()
    for tags_list in df['tags']:
        if isinstance(tags_list, list):
            all_tags.update(tags_list)

    tag_options = ['Все теги'] + sorted(list(all_tags))
    selected_view_tag = st.selectbox("Выберите тег для просмотра записей", tag_options, key="tag_select")
    n_records = st.number_input("Количество записей", min_value=10, max_value=500, value=100, step=10,
                                key="n_records")

    recent_records = get_recent_records_by_tag(df, selected_view_tag, n_records)

    if not recent_records.empty:
        st.dataframe(
            recent_records,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Дата и время": st.column_config.DatetimeColumn("Дата и время", format="DD.MM.YYYY HH:mm:ss"),
                "От кого": st.column_config.TextColumn("От кого"),
                "Краткое содержание": st.column_config.TextColumn("Краткое содержание", width="large"),
                "Теги": st.column_config.TextColumn("Теги"),
                "Исходный текст": st.column_config.TextColumn("Исходный текст"),
                "Имя файла": st.column_config.TextColumn("Имя файла")
            }
        )

        col_stats_tag1, col_stats_tag2, col_stats_tag3 = st.columns(3)

        with col_stats_tag1:
            st.metric("Показано записей", len(recent_records))

        with col_stats_tag2:
            if selected_view_tag != 'Все теги':
                total_with_tag = len(df[df['tags'].apply(lambda x: selected_view_tag in x)])
                st.metric("Всего с этим тегом", total_with_tag)

        with col_stats_tag3:
            date_range = f"{recent_records['Дата и время'].min()} - {recent_records['Дата и время'].max()}"
            st.metric("Диапазон дат", date_range)
    else:
        st.info(f"Нет записей с тегом '{selected_view_tag}'")


def filter_by_hot_tags(df):
    col_termination, col_prediction = st.columns([1, 1])
    set_prediction = {'col': col_prediction, 'name': 'AI RCT', 'tags': ['AI RCT']}
    termination_tags = ['расторжение договора', 'приостановить услуги', 'клиент недоволен и угрожает отказом от услуг']
    set_termination = {'col': col_termination, 'name': 'Расторжения и приостановки', 'tags': termination_tags}
    for s in [set_termination, set_prediction]:
        with s['col']:
            st.markdown(s['name'])

            last_month_df = filter_by_timeframe(df, 'Последний месяц')
            display_df = get_recent_records_by_tag(last_month_df, s['tags'])
            if not display_df.empty:
                filter_option = st.radio(
                    'Фильтр:',
                    ['Все', 'Непрочитанные'],
                    horizontal=True,
                    key=f'filter_{s["name"]}'
                )
            
                if filter_option == 'Непрочитанные':
                    display_df = display_df[~display_df['is_read']]
                display_df['is_read'] = display_df['is_read'].apply(lambda x: '🔴 Новое' if not x else '✅ Прочитано')
            
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )

                col_terms1, col_terms2 = st.columns(2)

                with col_terms1:
                    st.metric("Всего сообщений", len(display_df))

                with col_terms2:
                    unique_files = display_df[display_df['Источник'] != '']['Источник'].nunique()
                    st.metric("Уникальных отправителей", unique_files)
            else:
                st.info("Нет обращений за последний месяц")

def draw_graphs(df):
    TAGS_OF_INTEREST = ['клиент уходит к конкурентам', 'клиент недоволен и угрожает отказом от услуг', 'расторжение договора', 'клиент возмущен', 'mail']

    st.markdown("### Параметры фильтрации")
    timeframe = st.selectbox(
        "Выберите период",
        ['Все время', 'Последний месяц', 'Последний квартал', 'Последний год'],
        key="timeframe_select"
    )

    st.markdown("Выберите теги для отображения")
    selected_tags = []
    cols = st.columns(3)
    for idx, tag in enumerate(TAGS_OF_INTEREST):
        with cols[idx % 3]:
            if st.checkbox(tag, value=(False if tag=='mail' else True), key=f"tag_{tag}"):
                selected_tags.append(tag)

    filtered_df = filter_by_timeframe(df, timeframe)

    if selected_tags:
        tag_data = prepare_tag_data(filtered_df, selected_tags)

        if not tag_data.empty:
            fig = px.line(
                tag_data,
                x='date',
                y='count',
                color='tag',
                title=f'Динамика тегов за период: {timeframe}',
                labels={'date': 'Дата', 'count': 'Количество обращений', 'tag': 'Тег'}
            )

            fig.update_layout(
                xaxis_title="Дата",
                yaxis_title="Количество обращений",
                hovermode='x unified',
                legend_title_text='Теги'
            )

            fig.update_traces(mode='lines+markers')

            st.plotly_chart(fig, use_container_width=True)

            total_stats = tag_data.groupby('tag')['count'].sum().reset_index()
            total_stats.columns = ['Тег', 'Всего обращений']
            total_stats = total_stats.sort_values('Всего обращений', ascending=False)

            st.markdown("Статистика")
            col_stats1, col_stats2, col_stats3 = st.columns(3)

            with col_stats1:
                st.metric("Всего обращений", len(filtered_df))

            with col_stats2:
                st.metric("Уникальных тегов", len(selected_tags))

            with col_stats3:
                total_with_tags = len(filtered_df[filtered_df['tags'].apply(
                    lambda x: any(tag in x for tag in selected_tags)
                )])
                st.metric("Обращений с выбранными тегами", total_with_tags)

            st.dataframe(total_stats, use_container_width=True)
        else:
            st.warning("Нет данных для выбранных тегов в указанном периоде")

def ai_analyst(df):
    st.markdown("🤖  AI Аналитик")
    st.markdown("Задайте вопрос о данных в свободной форме")

    if not st.session_state.initialized:
        if st.button("Запустить AI аналитика", use_container_width=True):
            init_system()
            st.rerun()
        st.info("Нажмите кнопку выше, чтобы активировать AI аналитика")
    else:
        with st.expander("ℹ️ Информация о системе", expanded=False):
            if st.session_state.system:
                try:
                    info = st.session_state.system.get_system_info()
                    st.write(f"**Всего звонков:** {info.get('total_calls', 0)}")
                    st.write(f"**Уникальных тегов:** {info.get('unique_tags_count', 0)}")
                    if info.get('date_range', {}).get('start'):
                        start_date = datetime.fromisoformat(info['date_range']['start']).strftime('%d.%m.%Y')
                        end_date = datetime.fromisoformat(info['date_range']['end']).strftime('%d.%m.%Y')
                        st.write(f"**Период:** {start_date} - {end_date}")
                    st.write(f"**Модель:** {info.get('model', 'N/A')}")
                except:
                    st.write("Информация временно недоступна")

        #with st.expander("💡 Примеры запросов", expanded=False):
        #    examples = [
        #        "Какие самые частые темы обращений?",
        #        "Покажи динамику обращений по дням",
        #        "Какие проблемы требуют срочного решения?",
        #        "Сколько обращений с тегом 'расторжение договора'?"
        #    ]
        #    for example in examples:
        #        if st.button(f"📝 {example}", key=f"example_{example[:20]}", use_container_width=True):
        #            st.session_state.user_input = example
        #            st.rerun()

        #st.markdown("---")

        chat_container = st.container(height=400)

        with chat_container:
            if not st.session_state.messages:
                st.info("Задайте вопрос в поле ввода ниже")

            for message in st.session_state.messages:
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.write(message["content"])
                else:
                    with st.chat_message("assistant"):
                        st.write(message["content"])
                        if "stats" in message:
                            st.caption(f"📊 Проанализировано звонков: {message['stats']}")

        st.markdown("Ваш вопрос")
        user_input = st.chat_input("Напишите ваш вопрос здесь...", key="chat_input")

        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})

            with st.chat_message("user"):
                st.write(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Анализирую данные..."):
                    try:
                        st.session_state.query_history.append({
                            'query': user_input,
                            'timestamp': datetime.now(),
                            'status': 'processing'
                        })

                        result = st.session_state.system.process_query(
                            user_input,
                            st.session_state.query_history
                        )

                        st.session_state.query_history[-1]['status'] = 'completed'
                        st.session_state.query_history[-1]['result'] = result

                        st.write(result['answer'])
                        st.caption(f"Проанализировано звонков: {result.get('total_calls_analyzed', 0)}")

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result['answer'],
                            "stats": result.get('total_calls_analyzed', 0)
                        })

                        # Rerun to update chat display
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Ошибка: {e}")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Произошла ошибка: {e}"
                        })

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Очистить историю", use_container_width=True):
                st.session_state.messages = []
                st.session_state.query_history = []
                st.rerun()

        with col_btn2:
            if st.button("💾 Сохранить сессию", use_container_width=True):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"session_{timestamp}.json"
                os.makedirs(config['folders']['saved_results'], exist_ok=True)
                filepath = os.path.join(config['folders']['saved_results'], filename)

                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(st.session_state.query_history, f, ensure_ascii=False, indent=2, default=str)
                st.success(f"✅ Сохранено: {filename}")

# ==================== LEFT COLUMN - DASHBOARD ====================
with left_col:
    st.markdown("Дашборд аналитики")
    df = load_data(config['folders']['csv_mail'])
    print('Data loaded')
    try:
        filter_by_hot_tags(df)

        st.markdown("---")
        st.markdown("Просмотр последних записей по тегу")
        filter_by_selected_tags(df)
        draw_graphs(df)

        with st.expander("Просмотр исходных данных"):
            st.dataframe(df.head(100), use_container_width=True)

            st.markdown("### Статистика по всем тегам")
            all_tags_stats = []
            for tags_list in df['tags']:
                if isinstance(tags_list, list):
                    all_tags_stats.extend(tags_list)

            if all_tags_stats:
                tags_series = pd.Series(all_tags_stats)
                tags_stats = tags_series.value_counts().reset_index()
                tags_stats.columns = ['Тег', 'Количество']
                st.dataframe(tags_stats, use_container_width=True)

    except FileNotFoundError:
        st.error("Файл не найден. Пожалуйста, убедитесь, что файл существует.")
    except Exception as e:
        raise e

with right_col:
    ai_analyst(df)

