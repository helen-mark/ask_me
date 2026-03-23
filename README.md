The csv directories contain email and phonecall data. To start working with MCP system we need the raw data to be assigned tags.

Run process_csv_data.py to start preprocessing (assigning tags and short summary to each text). This process is to be scheduled to systematically preprocess all newly obtained raw texts.

Set llm model name in config.yml. Run main.py to start communication with MCP system. The system will address the preprocessed csv data, calculate metrics on tags and give a response.