import os
import yaml
from csv_tagger import CsvProcessor

def main():
    print("CSV PROCESSOR - LAUNCH")
    print("=" * 50)
    with open('config.yml', 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    processor = CsvProcessor(
        model=config["llm_model"],
        output_csv_path=os.path.join(config["folders"]["csv_mail"], 'unique_messages.csv'),
        batch_size=50,
        mail=True,
        config_path='config.yml'
    )

    print("\nStarting CSV tagging process...")
    processor.process()

    print(f"\n Processing finished!")
    print(f" Results saved to: {os.path.join(config['folders']['csv_mail'], 'unique_messages.csv')}")


if __name__ == "__main__":
    main()