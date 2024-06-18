#!/usr/bin/env python

import os
from art import text2art
from colorama import init, Fore, Style

import MySQLdb
import logging
import re
from bs4 import BeautifulSoup
import requests
from requests.auth import HTTPBasicAuth
import json
from transformers import (
    Trainer,
    TrainingArguments,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    DataCollatorForLanguageModeling,
)
import torch
from torch.optim import AdamW
from datasets import load_dataset, Dataset
from torch.cuda import OutOfMemoryError

# Init Logging
logging.basicConfig(level=logging.ERROR)

# Initialize colorama
init(autoreset=True)

# Clear the screen
os.system("cls" if os.name == "nt" else "clear")

# Print a fancy title using ANSI escape codes
# Generate the title using the art module
title_art = text2art("- cca-gpt-model-trainer -", font="tarty4")
indent = ":: "

# Print the title with color and shadow effect
print(Fore.GREEN + Style.BRIGHT + title_art)
print(
    Fore.CYAN
    + Style.BRIGHT
    + "\nVersion 1.0 // Written by Dean Thomson (grahfmusic) // github.com/grahfmusic \n"
)

# Ensure the output directory exists
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)


def fetch_mediawiki_data():
    """
    Fetches MediaWiki data and saves it to a text file.
    Change the host, user, password and database credentials - IMPORTANT
    """
    try:
        task = "Fetching MediaWiki SQL Data."
        print(Fore.YELLOW + indent + Fore.LIGHTCYAN_EX + task)
        connection = MySQLdb.connect(
            host="localhost", user="grahf", password="<password>", database="local_wiki"
        )
        cursor = connection.cursor()
        query = """
            SELECT page_title, old_text
            FROM page
            JOIN revision ON page.page_id = revision.rev_page
            JOIN text ON revision.rev_id = text.old_id;
        """
        cursor.execute(query)
        rows = cursor.fetchall()

        output_file = os.path.join(output_dir, "mediawiki_content.txt")
        with open(output_file, "w", encoding="utf-8") as file:
            for row in rows:
                file.write(f"{row[0]}\n{row[1]}\n")

        print(
            Fore.GREEN
            + indent
            + Fore.LIGHTCYAN_EX
            + "MediaWiki data fetched and saved."
        )
    except Exception as e:
        logging.error(f"Error fetching MediaWiki data: {e}")
        print(Fore.RED + f"Error fetching MediaWiki data: {e}")


def clean_mediawiki_data():
    """
    Cleans the MediaWiki data by removing HTML tags and other unnecessary content.
    """
    try:
        task = "Cleaning MediaWiki Data."
        print(Fore.YELLOW + indent + Fore.LIGHTCYAN_EX + task)

        input_file = os.path.join(output_dir, "mediawiki_content.txt")
        output_file = os.path.join(output_dir, "cleaned_mediawiki_content.txt")

        with open(input_file, "r", encoding="utf-8") as file:
            content = file.read()

        cleaned_content = re.sub(r"<[^>]+>", "", content)

        with open(output_file, "w", encoding="utf-8") as file:
            file.write(cleaned_content)

        print(Fore.GREEN + indent + Fore.LIGHTCYAN_EX + "MediaWiki data cleaned.")
    except Exception as e:
        logging.error(f"Error cleaning MediaWiki data: {e}")
        print(Fore.RED + f"Error cleaning MediaWiki data: {e}")


def fetch_jira_data():
    """
    Fetches Jira entries using the Jira REST API and saves them to a JSON file.
    """
    try:
        task = "Fetching Jira Entries from Jira Cloud."
        print(Fore.YELLOW + indent + Fore.LIGHTCYAN_EX + task)
        url = "https://site.atlassian.net/rest/api/3/search"  # CHANGE THIS TO APPROPRIATE JIRA URL
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        params = {
            "jql": "project = CSS",  # CHANGE THIS TO APPROPRIATE PROJECT CODE
            "maxResults": 3000,
            "fields": "summary,description,comment",
        }

        email = "email@email.com"  # CHANGE THIS TO APPROPRIATE JIRA USER
        api_token = "<token>"  # ADD JIRA TOKEN

        response = requests.get(
            url, headers=headers, params=params, auth=HTTPBasicAuth(email, api_token)
        )

        if response.status_code == 200:
            jira_data = response.json()
            output_file = os.path.join(output_dir, "jira_issues.json")
            with open(output_file, "w", encoding="utf-8") as file:
                json.dump(jira_data, file, indent=4)

            print(
                Fore.GREEN + indent + Fore.LIGHTCYAN_EX + "Jira data fetched and saved."
            )
        else:
            logging.error(f"Error fetching Jira data: HTTP {response.status_code}")
            print(Fore.RED + f"Error fetching Jira data: HTTP {response.status_code}")
    except Exception as e:
        logging.error(f"Error fetching Jira data: {e}")
        print(Fore.RED + f"Error fetching Jira data: {e}")


def combine_files(jira_file, mediawiki_file, combined_file):
    """
    Combines Jira and MediaWiki data into a single text file for training.
    """
    try:
        task = "Combining Jira and MediaWiki Data."
        print(Fore.YELLOW + indent + Fore.LIGHTCYAN_EX + task)

        with open(jira_file, "r", encoding="utf-8") as file:
            jira_data = json.load(file)

        jira_content = ""
        for issue in jira_data["issues"]:
            summary = issue["fields"]["summary"]
            description = issue["fields"].get("description", "")
            jira_content += f"{summary}\n{description}\n"

        with open(mediawiki_file, "r", encoding="utf-8") as file:
            mediawiki_content = file.read()

        with open(combined_file, "w", encoding="utf-8") as file:
            file.write(jira_content)
            file.write(mediawiki_content)

        print(Fore.GREEN + indent + Fore.LIGHTCYAN_EX + "Data combined for training.")
    except Exception as e:
        logging.error(f"Error combining data: {e}")
        print(Fore.RED + f"Error combining data: {e}")


def download_tokenizer_files(model_name, output_dir):
    """
    Downloads the tokenizer files for the specified model.
    """
    try:
        task = "Downloading Tokenizer Files."
        print(Fore.YELLOW + indent + Fore.LIGHTCYAN_EX + task)

        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token
        tokenizer.save_pretrained(output_dir)

        print(Fore.GREEN + indent + Fore.LIGHTCYAN_EX + "Tokenizer files downloaded.")
    except Exception as e:
        logging.error(f"Error downloading tokenizer files: {e}")
        print(Fore.RED + f"Error downloading tokenizer files: {e}")


def save_model_and_tokenizer(model, tokenizer, output_dir):
    # Save model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def fine_tune_gpt_model(data_file, output_dir):
    """
    Fine-tunes a GPT model on the combined data.
    """
    try:
        task = "Fine-Tuning GPT Model."
        print(Fore.YELLOW + indent + Fore.LIGHTCYAN_EX + task)

        model_name = "distilgpt2"
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token

        def tokenize_function(examples):
            return tokenizer(
                examples["text"], truncation=True, padding="max_length", max_length=512
            )

        data = load_dataset("text", data_files=data_file)
        tokenized_data = data.map(
            tokenize_function, batched=True, remove_columns=["text"]
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            warmup_steps=10,
            logging_dir=None,  # Disable logging directory
            logging_steps=10,  # Disable logging during training
            save_steps=500,
            save_total_limit=2,
            report_to=None,  # Disable reporting to any logging services like TensorBoard
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_data["train"],
        )

        try:
            trainer.train()
            save_model_and_tokenizer(model, tokenizer, output_dir)
        except OutOfMemoryError as e:
            logging.error(f"GPU out of memory: {e}")
            print(
                Fore.RED
                + "GPU out of memory. Consider reducing batch size, using gradient accumulation, or switching to CPU."
            )
            # Retry on CPU
            print(Fore.YELLOW + "Retrying on CPU...")
            training_args.no_cuda = True
            trainer.args = training_args
            trainer.train()
            save_model_and_tokenizer(model, tokenizer, output_dir)

    except Exception as e:
        logging.error(f"Error during model fine-tuning: {e}")
        print(Fore.RED + f"Error during model fine-tuning: {e}")


def main():
    fetch_mediawiki_data()
    clean_mediawiki_data()
    fetch_jira_data()

    jira_file = os.path.join(output_dir, "jira_issues.json")
    mediawiki_file = os.path.join(output_dir, "cleaned_mediawiki_content.txt")
    combined_file = os.path.join(output_dir, "combined_training_data.txt")
    combine_files(jira_file, mediawiki_file, combined_file)

    download_tokenizer_files("distilgpt2", output_dir)
    fine_tune_gpt_model(combined_file, output_dir)


if __name__ == "__main__":
    main()
